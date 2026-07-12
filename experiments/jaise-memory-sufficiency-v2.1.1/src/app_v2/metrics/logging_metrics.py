from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "power").mkdir(exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.steps_path = self.run_dir / "steps.jsonl"
        self.failures_path = self.run_dir / "failures.jsonl"

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def log_episode(
        self,
        episode: Dict[str, Any],
        steps: List[Dict[str, Any]],
        power_samples: List[Dict[str, Any]],
    ) -> None:
        self._append_jsonl(self.episodes_path, episode)
        for step in steps:
            self._append_jsonl(self.steps_path, step)

        if power_samples:
            output = self.run_dir / "power" / f"{episode['run_key']}.csv.gz"
            with gzip.open(
                output, "wt", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(power_samples[0].keys())
                )
                writer.writeheader()
                writer.writerows(power_samples)

    def log_failure(self, failure: Dict[str, Any]) -> None:
        self._append_jsonl(self.failures_path, failure)

    def completed_run_keys(self) -> Set[str]:
        if not self.episodes_path.exists():
            return set()
        keys: Set[str] = set()
        with self.episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("status") == "completed":
                        keys.add(str(record["run_key"]))
        return keys
