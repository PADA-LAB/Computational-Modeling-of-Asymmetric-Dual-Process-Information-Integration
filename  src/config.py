# src/config.py
from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_project_root() -> Path:
    # repo/main.py 기준 실행한다고 가정
    return Path(__file__).resolve().parents[1]
