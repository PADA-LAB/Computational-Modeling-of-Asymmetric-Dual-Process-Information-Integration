# src/main.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, get_project_root
from src.runner import run_platform_pipeline
from src.utils import log_print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/runner.yaml")
    parser.add_argument(
        "--platforms",
        type=str,
        default="Amazon,Coursera,Audible,Hotel",
        help="Comma-separated platform names.",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    cfg_path = project_root / args.config
    cfg = load_config(cfg_path)

    cfg["_project_root"] = str(project_root.resolve())

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]

    log_print(f"[MAIN] project_root = {project_root}")
    log_print(f"[MAIN] config       = {cfg_path}")
    log_print(f"[MAIN] platforms    = {platforms}")

    for platform in platforms:
        run_platform_pipeline(cfg=cfg, platform=platform, project_root=project_root)

    log_print("[MAIN] ALL DONE")


if __name__ == "__main__":
    main()