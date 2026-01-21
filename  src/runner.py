# src/runner.py (함수 시그니처 + 단계 분기만 반영)
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .utils import (
    log_print, ensure_dir, get_s1_feature_cols,
    resolve_cfg_paths_abs, should_use_finetuned_t5,
)

from .finetune_t5_lora import run_lora_finetune
from .s1 import run_s1_train_and_save
from .s2 import run_s2_train_and_save
from .early_fusion import run_early_fusion
from .residual import run_residual_both_directions
from .late_fusion import run_late_fusion

_ALLOWED_STAGES = {"s1", "s2", "early-fusion", "residual", "late-fusion"}


def run_platform_pipeline(*, cfg: dict, platform: str, project_root: Path, stages: list[str]):
    # stage validate
    unknown = [s for s in stages if s not in _ALLOWED_STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Allowed: {sorted(_ALLOWED_STAGES)}")

    paths_abs = resolve_cfg_paths_abs(cfg, project_root)

    data_map = cfg["paths"]["data"]
    csv_rel = data_map.get(platform)
    if not csv_rel:
        raise KeyError(f"paths.data에 {platform}이 없습니다. (config 확인)")

    csv_path = (project_root / csv_rel).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"missing data: {csv_path}")

    out_root = Path(paths_abs["outputs_root"])
    platform_out_dir = ensure_dir(out_root / platform)

    cols = cfg["columns"]
    text_col = cols["text"]
    y_col = cols["label"]

    exp = cfg["experiment"]
    seeds = exp["seeds"]
    n_trials_s1 = exp["n_trials_s1"]
    n_trials_s2 = exp["n_trials_s2"]
    s1_models = exp["s1_models"]
    s2_models = exp["s2_models"]
    resid_s2_models = exp["resid_s2_models"]
    resid_s1_models = exp["resid_s1_models"]
    batch_size = exp.get("embedding_batch_size", 32)

    s1_feature_cols = get_s1_feature_cols(cfg, platform)

    log_print("=" * 80)
    log_print(f"=== Platform: {platform} ===")
    log_print(f"Data: {csv_path}")
    log_print(f"Stages: {stages}")
    log_print(f"S1 Features({len(s1_feature_cols)}): {s1_feature_cols}")
    log_print(f"Outputs: {platform_out_dir}")
    log_print("=" * 80)

    df = pd.read_csv(csv_path)

    # LoRA finetune은 "s2"가 포함될 때만 필요 (임베딩 만들 때만)
    if "s2" in stages and should_use_finetuned_t5(cfg):
        ft = cfg["finetune"]
        lora_dir = Path(paths_abs["finetune_model_dir"]) / platform

        if not lora_dir.exists() or (
            not (lora_dir / "adapter_model.bin").exists()
            and not (lora_dir / "adapter_model.safetensors").exists()
        ):
            log_print(f"[LoRA] adapter not found -> finetune start: {platform}")
            run_lora_finetune(
                platform=platform,
                df=df,
                text_col=text_col,
                label_col=y_col,
                model_name=ft["model_name"],
                max_length=ft["max_length"],
                target_seed_for_t5=ft["target_seed_for_t5"],
                ckpt_dir=Path(paths_abs["finetune_ckpt_dir"]),
                final_model_dir=Path(paths_abs["finetune_model_dir"]),
                lora_r=ft["lora"]["r"],
                lora_alpha=ft["lora"]["alpha"],
                lora_dropout=ft["lora"]["dropout"],
                train_epochs=ft["train"]["epochs"],
                lr=ft["train"]["lr"],
                per_device_train_bs=ft["train"]["per_device_train_batch_size"],
                per_device_eval_bs=ft["train"]["per_device_eval_batch_size"],
                weight_decay=ft["train"]["weight_decay"],
            )
        else:
            log_print(f"[LoRA] adapter exists -> skip: {lora_dir}")

    for seed in seeds:
        seed_dir = ensure_dir(platform_out_dir / f"seed_{seed}")

        # Step1: S1
        if "s1" in stages:
            if not (seed_dir / "done_s1.flag").exists():
                run_s1_train_and_save(
                    df=df,
                    s1_feature_cols=s1_feature_cols,
                    y_col=y_col,
                    seed=seed,
                    s1_models=s1_models,
                    n_trials=n_trials_s1,
                    seed_dir=seed_dir,
                )

        # Step2: S2
        if "s2" in stages:
            run_s2_train_and_save(
                cfg=cfg,
                platform=platform,
                df=df,
                text_col=text_col,
                y_col=y_col,
                seed=seed,
                s2_models=s2_models,
                n_trials=n_trials_s2,
                seed_dir=seed_dir,
                platform_out_dir=platform_out_dir,
                batch_size=batch_size,
            )

        # Step3: Early Fusion (S1+S2 둘 다 있어야 의미 있음)
        if "early-fusion" in stages:
            run_early_fusion(
                cfg=cfg,
                df=df,
                s1_feature_cols=s1_feature_cols,
                text_col=text_col,
                y_col=y_col,
                seed=seed,
                seed_dir=seed_dir,
                platform_out_dir=platform_out_dir,
                n_trials=n_trials_s2,
            )

        # Step4: Residual (보통 S1/S2 결과가 있어야 함)
        if "residual" in stages:
            run_residual_both_directions(
                cfg=cfg,
                df=df,
                s1_feature_cols=s1_feature_cols,
                text_col=text_col,
                y_col=y_col,
                seed=seed,
                seed_dir=seed_dir,
                platform_out_dir=platform_out_dir,
                n_trials=n_trials_s2,
                s1_models=resid_s1_models,
                resid_s2_models=resid_s2_models,
            )

        # Step5: late fusion
        if "late-fusion" in stages:
            # late-fusion은 기존 산출물(s1 csv + s2 pkl)이 필요하므로, 최소 체크
            missing = []

            if not (seed_dir / "train_idx.npy").exists():
                missing.append("train_idx.npy")
            if not (seed_dir / "test_idx.npy").exists():
                missing.append("test_idx.npy")

            # S2 결과 (joblib pkl)
            if not (seed_dir / "s2_preds_dict.pkl").exists():
                missing.append("s2_preds_dict.pkl")

            # S1 결과 (각 모델별 csv)
            for s1_name in s1_models:
                s1_dir = seed_dir / f"S1_{s1_name}"
                if not (s1_dir / "s1_train_preds.csv").exists():
                    missing.append(f"S1_{s1_name}/s1_train_preds.csv")
                if not (s1_dir / "s1_pred_proba.csv").exists():
                    missing.append(f"S1_{s1_name}/s1_pred_proba.csv")

            if missing:
                log_print(f"[LateFusion] skip seed={seed} (missing files): {missing}")
            else:
                run_late_fusion(
                    cfg=cfg,
                    df=df,
                    y_col=y_col,
                    seed=seed,
                    seed_dir=seed_dir,
                    platform_out_dir=platform_out_dir,  # 인터페이스 맞추기용 (late_fusion에서 현재 미사용)
                    s1_models=s1_models,
                    s2_models=s2_models,
                )

    log_print(f"[DONE] {platform}")