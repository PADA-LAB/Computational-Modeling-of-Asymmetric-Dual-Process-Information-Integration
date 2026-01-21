# review_helpfulness_runner/main.py
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from .config import RunnerConfig
from .utils import get_project_root, resolve_data_path, make_output_dir, ensure_dir, log_print
from .s1 import run_s1_train_and_save
from .s2 import run_s2_train_and_save
from .early_fusion import run_early_fusion
from .residual import run_residual_both_directions
from .report import build_report


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/runner.yaml")
    ap.add_argument("--platform", type=str, default="Amazon", help="Amazon|Hotel|Coursera|Audible|ALL")
    ap.add_argument("--project-root", type=str, default=None, help="repo root. If omitted, auto-detect.")
    ap.add_argument("--skip-residual", action="store_true")
    ap.add_argument("--skip-earlyfusion", action="store_true")
    return ap.parse_args()


def run_one_platform(cfg: RunnerConfig, platform: str, project_root: Path):
    exp = cfg.experiment
    out_root_rel = cfg.output["root_dir"]

    data_path = resolve_data_path(project_root, cfg.data[platform])
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    out_dir = make_output_dir(project_root, out_root_rel, platform)

    df = pd.read_csv(data_path)

    s1_feature_cols = cfg.s1_features[platform]
    y_col = exp["label_column"]
    text_col = exp["text_column"]

    seeds = exp["seeds"]
    s1_models = exp["s1_models"]
    s2_models = exp["s2_models"]
    n_trials_s1 = exp["n_trials_s1"]
    n_trials_s2 = exp["n_trials_s2"]
    embedding_model = exp["embedding_model"]
    batch_size = exp.get("batch_size", 32)

    for seed in seeds:
        seed_dir = ensure_dir(out_dir / f"seed_{seed}")

        # S1
        if not (seed_dir / "done_s1.flag").exists():
            log_print(f"[{platform}] Seed {seed}: Running S1...")
            run_s1_train_and_save(
                df=df,
                s1_feature_cols=s1_feature_cols,
                y_col=y_col,
                seed=seed,
                s1_models=s1_models,
                n_trials=n_trials_s1,
                seed_dir=seed_dir,
            )

        # S2
        log_print(f"[{platform}] Seed {seed}: Running S2 (emb={embedding_model})...")
        run_s2_train_and_save(
            df=df,
            text_col=text_col,
            y_col=y_col,
            seed=seed,
            s2_models=s2_models,
            n_trials=n_trials_s2,
            seed_dir=seed_dir,
            platform_out_dir=out_dir,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )

        # Early Fusion
        if not exp.get("disable_early_fusion", False):
            log_print(f"[{platform}] Seed {seed}: Running Early Fusion...")
            run_early_fusion(
                df=df,
                s1_feature_cols=s1_feature_cols,
                text_col=text_col,
                y_col=y_col,
                seed=seed,
                seed_dir=seed_dir,
                platform_out_dir=out_dir,
                embedding_model=embedding_model,
                batch_size=batch_size,
                n_trials=n_trials_s2,
            )

        # Residual
        if not exp.get("disable_residual", False):
            log_print(f"[{platform}] Seed {seed}: Running Residual...")
            run_residual_both_directions(
                df=df,
                s1_feature_cols=s1_feature_cols,
                text_col=text_col,
                y_col=y_col,
                seed=seed,
                seed_dir=seed_dir,
                platform_out_dir=out_dir,
                embedding_model=embedding_model,
                batch_size=batch_size,
                n_trials=n_trials_s2,
                s1_models=s1_models,
                resid_s2_models=s2_models,
            )

    # Report
    build_report(out_dir, platform, seeds=seeds, s1_models=s1_models)
    log_print(f"[{platform}] Done. Outputs at: {out_dir}")


def main():
    args = parse_args()
    runner_dir = Path(__file__).resolve().parent
    project_root = get_project_root(args.project_root, runner_dir)

    cfg_path = (project_root / args.config).resolve()
    cfg = RunnerConfig.load(cfg_path)

    if args.platform == "ALL":
        for platform in cfg.data.keys():
            run_one_platform(cfg, platform, project_root)
    else:
        run_one_platform(cfg, args.platform, project_root)


if __name__ == "__main__":
    main()