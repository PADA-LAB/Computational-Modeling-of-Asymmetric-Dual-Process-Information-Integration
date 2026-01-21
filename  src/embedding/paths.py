# src/embedding/paths.py
from pathlib import Path
from datetime import datetime

def resolve_data_path(project_root: Path, relative_path: str) -> Path:
    return (project_root / relative_path).resolve()

def make_output_path(project_root: Path, out_root_rel: str, platform: str) -> Path:
    out_root = (project_root / out_root_rel).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # timestamp를 넣고 싶으면 아래 주석 해제
    # run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_dir = out_root / platform / run_id
    out_dir = out_root / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir