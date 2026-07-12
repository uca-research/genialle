from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from app_v2.config import BASE_DIR, SETTINGS_PATH, TASKS_PATH, load_settings
from app_v2.llm_client import LLMClient
from app_v2.memory import (
    LexicalRetrievalMemory,
    NoMemory,
    PerAgentMemory,
    ResponsibleLearnerStateMemory,
    SharedMemory,
)
from app_v2.metrics.energy import NVMLPowerMonitor
from app_v2.metrics.logging_metrics import RunLogger
from app_v2.orchestrator.episode_runner import EpisodeRunner
from app_v2.tasks_loader import load_tasks


def make_memory(condition: str, settings: Dict[str, Any]) -> Any:
    cfg = settings["memory"]
    if condition == "C0":
        return NoMemory()
    if condition == "C1":
        return PerAgentMemory(int(cfg["c1_max_chars"]))
    if condition == "C2":
        return SharedMemory(int(cfg["c2_max_chars"]))
    if condition == "C3":
        return LexicalRetrievalMemory(
            int(cfg["c3_top_k"]), int(cfg["c3_max_chars"])
        )
    if condition == "C4":
        return ResponsibleLearnerStateMemory(int(cfg["c4_max_chars"]))
    raise ValueError(f"Unknown condition: {condition}")


def build_schedule(
    conditions: List[str],
    tasks: List[Dict[str, Any]],
    repetitions: int,
    master_seed: int,
) -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        block = [
            {
                "condition": condition,
                "task_index": task_index,
                "task_id": task["id"],
                "repetition": repetition,
                "run_key": f"r{repetition:02d}-{task['id']}-{condition}",
            }
            for condition in conditions
            for task_index, task in enumerate(tasks)
        ]
        random.Random(master_seed + repetition).shuffle(block)
        schedule.extend(block)
    return schedule


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in ("requests", "PyYAML", "nvidia-ml-py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(BASE_DIR), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def write_manifest(
    run_dir: Path,
    mode: str,
    repetitions: int,
    settings: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    schedule: List[Dict[str, Any]],
    client: LLMClient,
    monitor: NVMLPowerMonitor,
) -> None:
    shutil.copy2(SETTINGS_PATH, run_dir / "settings_v2.yaml")
    shutil.copy2(TASKS_PATH, run_dir / "tasks_v2.yaml")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "repetitions": repetitions,
        "settings": settings,
        "task_ids": [task["id"] for task in tasks],
        "schedule": schedule,
        "server_health": client.health(),
        "server_models": client.models(),
        "gpu": monitor.device_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": git_value("status", "--porcelain"),
        },
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["smoke", "pilot", "full"], default="smoke"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    all_tasks = load_tasks()
    experiment = settings["experiment"]
    conditions = args.conditions or list(experiment["conditions"])

    invalid = set(conditions) - {"C0", "C1", "C2", "C3", "C4"}
    if invalid:
        raise ValueError(f"Invalid conditions: {sorted(invalid)}")

    tasks = all_tasks
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [task for task in all_tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise ValueError(f"Unknown task ids: {sorted(missing)}")
    elif args.mode == "smoke":
        tasks = all_tasks[:1]

    if args.repetitions is not None:
        repetitions = args.repetitions
    else:
        repetitions = int(experiment[f"{args.mode}_repetitions"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        BASE_DIR
        / str(settings["logging"]["root_dir"])
        / f"{args.mode}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir)

    client = LLMClient(settings)
    monitor = NVMLPowerMonitor(
        int(settings["energy"]["gpu_index"]),
        float(settings["energy"]["sample_interval_sec"]),
    )
    schedule = build_schedule(
        conditions,
        tasks,
        repetitions,
        int(experiment["master_seed"]),
    )
    write_manifest(
        run_dir,
        args.mode,
        repetitions,
        settings,
        tasks,
        schedule,
        client,
        monitor,
    )

    completed = logger.completed_run_keys() if args.resume else set()
    warmup_calls = int(experiment.get("warmup_calls", 0))
    if warmup_calls:
        print(f"Executing {warmup_calls} unlogged warm-up calls...")
        client.warmup(warmup_calls)

    print(f"RUN_DIR={run_dir}")
    print(f"Scheduled episodes: {len(schedule)}")
    failures = 0

    try:
        for position, item in enumerate(schedule, 1):
            if item["run_key"] in completed:
                print(f"[{position}/{len(schedule)}] SKIP {item['run_key']}")
                continue

            cooldown = float(experiment.get("cooldown_sec", 0.0))
            if cooldown:
                time.sleep(cooldown)

            task = next(t for t in tasks if t["id"] == item["task_id"])
            runner = EpisodeRunner(
                client,
                make_memory(item["condition"], settings),
                monitor,
                settings,
            )
            print(f"[{position}/{len(schedule)}] START {item['run_key']}")
            try:
                result = runner.run(
                    run_key=item["run_key"],
                    condition=item["condition"],
                    task=task,
                    task_index=int(item["task_index"]),
                    repetition=int(item["repetition"]),
                )
                logger.log_episode(
                    result["episode"],
                    result["steps"],
                    result["power_samples"],
                )
                episode = result["episode"]
                print(
                    f"[{position}/{len(schedule)}] DONE {item['run_key']} "
                    f"time={episode['wall_time_sec']:.2f}s "
                    f"tokens={episode['usage']['total_tokens']} "
                    f"dynamic_energy="
                    f"{episode['power']['dynamic_energy_j']:.1f}J "
                    f"truncated="
                    f"{episode['robustness']['truncated_calls']}"
                )
            except Exception as exc:
                failures += 1
                logger.log_failure(
                    {
                        **item,
                        "status": "failed",
                        "timestamp_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"FAILED {item['run_key']}: {exc}", file=sys.stderr)
                if not bool(
                    experiment.get("continue_after_failure", True)
                ):
                    raise
    finally:
        monitor.close()

    print(f"Experiment finished. Failures: {failures}")
    print(f"RUN_DIR={run_dir}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
