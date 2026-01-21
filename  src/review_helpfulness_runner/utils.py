# review_helpfulness_runner/utils.py
from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_project_root(project_root: Optional[str], runner_dir: Path) -> Path:
    """
    - project_root 인자가 있으면 그걸 사용
    - 없으면 runner_dir의 부모(= repo 루트)를 project_root로 사용
    """
    if project_root:
        return Path(project_root).resolve()
    return runner_dir.parent.resolve()


def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    return (project_root / relative_path).resolve()


def make_output_dir(project_root: Path, out_root_rel: str, platform: str) -> Path:
    out_root = (project_root / out_root_rel).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_print(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)