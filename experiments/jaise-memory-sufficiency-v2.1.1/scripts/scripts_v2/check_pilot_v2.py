from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    episodes = read_jsonl(args.run_dir / "episodes.jsonl")
    steps = read_jsonl(args.run_dir / "steps.jsonl")
    failures = read_jsonl(args.run_dir / "failures.jsonl")
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected = len(manifest["schedule"])
    agents = defaultdict(list)
    for step in steps:
        agents[str(step["agent"])].append(step)

    checks = []
    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    add(
        "All scheduled episodes completed",
        len(episodes) == expected and not failures,
        f"completed={len(episodes)}, expected={expected}, failures={len(failures)}",
    )

    empty = sum(bool(step.get("empty_output")) for step in steps)
    add(
        "Non-empty final content >= 99%",
        percent(len(steps) - empty, len(steps)) >= 99.0,
        f"empty={empty}/{len(steps)} ({percent(empty, len(steps)):.2f}%)",
    )

    markers = sum(
        bool(step.get("raw_reasoning_markers")) for step in steps
    )
    add(
        "No raw reasoning markers stored",
        markers == 0,
        f"marker_outputs={markers}",
    )

    evaluator_steps = agents.get("Evaluator", [])
    valid_evaluator = sum(
        bool(step.get("evaluator_json_valid")) for step in evaluator_steps
    )
    evaluator_valid_rate = percent(valid_evaluator, len(evaluator_steps))
    add(
        "Evaluator JSON validity >= 99%",
        evaluator_valid_rate >= 99.0,
        f"valid={valid_evaluator}/{len(evaluator_steps)} "
        f"({evaluator_valid_rate:.2f}%)",
    )

    for agent, agent_steps in sorted(agents.items()):
        truncated = sum(bool(step.get("truncated")) for step in agent_steps)
        rate = percent(truncated, len(agent_steps))
        add(
            f"{agent} truncation < 2%",
            rate < 2.0,
            f"truncated={truncated}/{len(agent_steps)} ({rate:.2f}%)",
        )

    insufficient_power = sum(
        int(ep.get("power", {}).get("sample_count", 0)) < 2
        for ep in episodes
    )
    nonpositive_baseline = sum(
        float(ep.get("baseline", {}).get("baseline_w_median", 0.0)) <= 0
        for ep in episodes
    )
    add(
        "Valid power traces",
        insufficient_power == 0 and nonpositive_baseline == 0,
        f"insufficient_samples={insufficient_power}, "
        f"nonpositive_baseline={nonpositive_baseline}",
    )

    contaminated = sum(
        int(ep.get("memory", {}).get("contains_current_round_reads", 0))
        for ep in episodes
    )
    add(
        "No current-round memory contamination",
        contaminated == 0,
        f"contaminated_reads={contaminated}",
    )

    deletion_failures = sum(
        not bool(ep.get("memory", {}).get("deletion_success"))
        for ep in episodes
    )
    add(
        "Episode memory deletion succeeded",
        deletion_failures == 0,
        f"deletion_failures={deletion_failures}",
    )

    retries = sum(
        int(ep.get("robustness", {}).get("request_retries", 0))
        for ep in episodes
    )
    conditions = Counter(ep["condition"] for ep in episodes)
    report = {
        "run_dir": str(args.run_dir),
        "mode": manifest["mode"],
        "expected_episodes": expected,
        "completed_episodes": len(episodes),
        "steps": len(steps),
        "failures": len(failures),
        "conditions": dict(conditions),
        "total_retries": retries,
        "checks": checks,
        "overall_pass": all(item["passed"] for item in checks),
    }

    json_path = args.run_dir / "acceptance_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Pilot acceptance report",
        "",
        f"- Run: `{args.run_dir}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Completed episodes: {len(episodes)}/{expected}",
        f"- Logged calls: {len(steps)}",
        f"- Failures: {len(failures)}",
        f"- Overall result: **{'PASS' if report['overall_pass'] else 'REVIEW'}**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        result = "PASS" if item["passed"] else "REVIEW"
        lines.append(
            f"| {item['check']} | {result} | {item['detail']} |"
        )
    markdown_path = args.run_dir / "acceptance_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(markdown_path.read_text(encoding="utf-8"))
    print(f"JSON report: {json_path}")
    raise SystemExit(0 if report["overall_pass"] else 2)


if __name__ == "__main__":
    main()
