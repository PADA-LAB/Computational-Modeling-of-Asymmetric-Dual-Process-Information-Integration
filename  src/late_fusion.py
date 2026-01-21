# src/late_fusion.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import average_precision_score

from .utils import ensure_dir, save_json, log_print, compute_metrics, find_best_threshold


def _weight_grid(step: float = 0.05) -> np.ndarray:
    # 0.0 ~ 1.0 inclusive
    n = int(round(1.0 / step))
    return np.linspace(0.0, 1.0, n + 1)


def run_late_fusion(
    *,
    cfg: dict,
    df: pd.DataFrame,
    y_col: str,
    seed: int,
    seed_dir: Path,
    platform_out_dir: Path,  # 인터페이스 맞추기용 (현재는 사용 안 해도 됨)
    s1_models: list[str],
    s2_models: list[str],
):
    """
    Late Fusion (Weighted Sum)
    - weight는 TRAIN에서 PR-AUC 최대가 되도록 선택 (leakage 방지)
    - threshold는 TRAIN에서 F1 최대 threshold 선택
    - TEST는 최종 평가만
    저장:
      seed_dir/Late_Fusion/<s2>__<s1>/metrics.json
      seed_dir/Late_Fusion/<s2>__<s1>/pred_late_fusion.npy
    """
    lf_root = ensure_dir(seed_dir / "Late_Fusion")

    # split
    tr_idx = np.load(seed_dir / "train_idx.npy", allow_pickle=False)
    te_idx = np.load(seed_dir / "test_idx.npy", allow_pickle=False)
    y_all = df[y_col].values.astype(int)
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    # S2 preds dict
    s2_res_path = seed_dir / "s2_preds_dict.pkl"
    if not s2_res_path.exists():
        log_print(f"[LateFusion] skip (missing): {s2_res_path}")
        return
    s2_preds = joblib.load(s2_res_path)  # {"train":{...}, "test":{...}}

    # weight grid
    lf_cfg = (cfg.get("late_fusion", {}) or {})
    step = float(lf_cfg.get("weight_step", 0.05))
    weights = _weight_grid(step)

    # (옵션) overwrite
    overwrite = bool(lf_cfg.get("overwrite", False))

    for s2_name in s2_models:
        if s2_name not in s2_preds.get("train", {}) or s2_name not in s2_preds.get("test", {}):
            continue

        p_s2_tr = np.asarray(s2_preds["train"][s2_name], dtype=np.float64)
        p_s2_te = np.asarray(s2_preds["test"][s2_name], dtype=np.float64)

        for s1_name in s1_models:
            # S1 train/test proba CSV 로드
            s1_dir = seed_dir / f"S1_{s1_name}"
            s1_tr_csv = s1_dir / "s1_train_preds.csv"
            s1_te_csv = s1_dir / "s1_pred_proba.csv"
            if not s1_tr_csv.exists() or not s1_te_csv.exists():
                continue

            p_s1_tr = pd.read_csv(s1_tr_csv)["s1_pred_proba"].values.astype(np.float64)
            p_s1_te = pd.read_csv(s1_te_csv)["s1_pred_proba"].values.astype(np.float64)

            save_dir = ensure_dir(lf_root / f"{s2_name}__{s1_name}")

            metrics_path = save_dir / "metrics.json"
            pred_path = save_dir / "pred_late_fusion.npy"

            if (not overwrite) and metrics_path.exists() and pred_path.exists():
                continue

            # --- 1) weight 선택: TRAIN에서 PR-AUC 최대 ---
            best_w = 0.5
            best_pr = -1.0
            best_tr_ens = None

            for w in weights:
                ens_tr = w * p_s2_tr + (1.0 - w) * p_s1_tr
                try:
                    pr = average_precision_score(y_tr, ens_tr)
                except Exception:
                    pr = -1.0

                if pr > best_pr:
                    best_pr = pr
                    best_w = float(w)
                    best_tr_ens = ens_tr

            if best_tr_ens is None:
                continue

            # --- 2) threshold 선택: TRAIN에서 F1 최대 ---
            t_opt, best_f1 = find_best_threshold(y_tr, best_tr_ens, metric="f1")

            # --- 3) TEST 평가 ---
            ens_te = best_w * p_s2_te + (1.0 - best_w) * p_s1_te
            ens_te = np.clip(ens_te, 0.0, 1.0)

            m = compute_metrics(y_te, ens_te, threshold=t_opt)
            m.update(
                {
                    "seed": int(seed),
                    "method": "Late Fusion (PR-AUC opt on TRAIN)",
                    "s2_model": f"S2_{s2_name}",
                    "s1_model": f"S1_{s1_name}",
                    "best_weight_s2": float(best_w),
                    "best_weight_s1": float(1.0 - best_w),
                    "train_best_pr_auc": float(best_pr),
                    "train_best_f1": float(best_f1),
                    "threshold_selected_on": "train",
                    "weight_selected_on": "train",
                    "weight_step": float(step),
                }
            )

            save_json(m, metrics_path)
            np.save(pred_path, ens_te.astype(np.float32))

            log_print(
                f"[LateFusion] seed={seed} {s2_name}(w={best_w:.2f}) + {s1_name}(w={1-best_w:.2f}) "
                f"TEST PR-AUC={m.get('pr_auc', np.nan):.4f}"
            )