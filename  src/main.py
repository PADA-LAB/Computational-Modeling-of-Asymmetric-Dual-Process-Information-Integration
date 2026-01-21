# src/main.py
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import load_config, get_project_root, log_print
from .runner import run_platform_pipeline

_ALLOWED_STAGES = {"s1", "s2", "early-fusion", "residual", "late-fusion"}
_ALL_STAGES_ORDERED = ["s1", "s2", "early-fusion", "residual", "late-fusion"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/runner.yaml")
    parser.add_argument(
        "--platforms",
        type=str,
        default="Amazon,Coursera,Audible,Hotel",
        help="Comma-separated platform names. e.g., Amazon,Hotel",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help=(
            "Comma-separated stages. e.g., s1 or s1,s2. "
            "Allowed: s1,s2,early-fusion,residual,late-fusion or 'all'"
        ),
    )
    args = parser.parse_args()

    project_root = get_project_root()
    cfg_path = (project_root / args.config).resolve()

    # (1) config 존재 체크
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)

    # project_root를 cfg에 주입 (절대경로 resolve에 사용)
    cfg["_project_root"] = str(project_root.resolve())

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    # (2) --stages all 지원
    if len(stages) == 1 and stages[0].lower() == "all":
        stages = list(_ALL_STAGES_ORDERED)

    # (3) stages 입력 검증 (main에서 먼저)
    unknown = [s for s in stages if s not in _ALLOWED_STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Allowed: {sorted(_ALLOWED_STAGES)} or 'all'")

    log_print(f"[MAIN] project_root = {project_root}")
    log_print(f"[MAIN] config       = {cfg_path}")
    log_print(f"[MAIN] platforms    = {platforms}")
    log_print(f"[MAIN] stages       = {stages}")

    for platform in platforms:
        run_platform_pipeline(
            cfg=cfg,
            platform=platform,
            project_root=project_root,
            stages=stages,
        )

    log_print("[MAIN] ALL DONE")


if __name__ == "__main__":
    main()