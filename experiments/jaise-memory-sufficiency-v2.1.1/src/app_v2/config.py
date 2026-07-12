from pathlib import Path
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = BASE_DIR / "config" / "settings_v2.yaml"
TASKS_PATH = BASE_DIR / "config" / "tasks_v2.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_settings(path: Path = SETTINGS_PATH) -> Dict[str, Any]:
    return load_yaml(path)
