# src/config.py
from __future__ import annotations

from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """
    src/config.py 기준:
    - .../<repo>/src/config.py
    - parents[1] == .../<repo>
    """
    return Path(__file__).resolve().parents[1]