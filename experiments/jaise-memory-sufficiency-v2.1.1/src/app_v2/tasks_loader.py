from pathlib import Path
from typing import Any, Dict, List

from app_v2.config import TASKS_PATH, load_yaml


def load_tasks(path: Path = TASKS_PATH) -> List[Dict[str, Any]]:
    data = load_yaml(path)
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"No tasks found in {path}")
    for task in tasks:
        rounds = task.get("rounds", [])
        if len(rounds) != 3:
            raise ValueError(
                f"Task {task.get('id')} must contain exactly three rounds"
            )
    return tasks
