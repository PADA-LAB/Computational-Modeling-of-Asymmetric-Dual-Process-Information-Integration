# src/early_fusion.py
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import joblib
import torch

from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils.torch_utils import SimpleMLP, train_torch, predict_torch
from .utils import save_json, log_print

# S2와 동일한 공통 캐시 파일을 로드
from .embedding.cache_utils import embedding_cache_file


def run_early_fusion(
    *,
    cfg: dict,
    df: pd.DataFrame,
    s1_feature_cols: list[str],
    text_col: str,
    y_col: str,
    seed: int,
    seed_dir: Path,
    platform_out_dir: Path,
    n_trials: int,
):
    ef_path = seed_dir / "early_fusion_preds.pkl"
    if ef_path.exists():
        return

    tr_idx = np.load(seed_dir / "train_idx.npy")
    te_idx = np.load(seed_dir / "test_idx.npy")

    y_all = df[y_col].values.astype(int)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    # S1 feature: 안전하게 numeric으로 캐스팅 (혹시 문자열/결측 섞이면 NaN 생김)
    use_cols = [c for c in s1_feature_cols if c in df.columns]
    X_s1 = df[use_cols].apply(pd.to_numeric, errors="coerce").values
    # NaN이 있으면 RobustScaler에 영향주므로 0으로 대체(혹은 median impute도 가능)
    X_s1 = np.nan_to_num(X_s1, nan=0.0, posinf=0.0, neginf=0.0)

    X_s1_tr, X_s1_te = X_s1[tr_idx], X_s1[te_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # S2 cache 재사용 (finetune/t5_lora 포함 동일 임베딩 보장)
    cache_path = embedding_cache_file(platform_out_dir, cfg)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"[EarlyFusion] Embedding cache not found: {cache_path}\n"
            f"S2를 먼저 실행해서 X_all.npy 캐시를 만든 뒤 EarlyFusion을 실행해야 해."
        )

    X_s2_all = np.load(cache_path, allow_pickle=False)

    # 캐시가 df와 같은 row-order/row-count인지 검증
    if X_s2_all.shape[0] != len(df):
        raise ValueError(
            f"[EarlyFusion] Embedding cache row mismatch.\n"
            f" - cache rows: {X_s2_all.shape[0]}\n"
            f" - df rows   : {len(df)}\n"
            f"같은 데이터/같은 전처리/같은 df 순서로 만든 캐시인지 확인해줘."
        )

    X_s2_tr, X_s2_te = X_s2_all[tr_idx], X_s2_all[te_idx]

    scaler = RobustScaler()
    X_ef_tr = np.hstack([scaler.fit_transform(X_s1_tr), X_s2_tr])
    X_ef_te = np.hstack([scaler.transform(X_s1_te), X_s2_te])

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def obj(trial):
        nl = trial.suggest_int("nl", 1, 3)
        nu = trial.suggest_int("nu", 64, 256)
        do = trial.suggest_float("do", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = []
        for t, v in skf.split(X_ef_tr, y_tr):
            m = SimpleMLP(X_ef_tr.shape[1], nl, nu, do)
            m = train_torch(m, X_ef_tr[t], y_tr[t], device=device, lr=lr, epochs=5)
            p = predict_torch(m, X_ef_tr[v], device=device)
            scores.append(average_precision_score(y_tr[v], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials)

    save_json(
        {
            "seed": seed,
            "model": "EarlyFusion_MLP",
            "best_score": study.best_value,
            "best_params": study.best_params,
            "embedding_cache_file": str(cache_path),
            "finetune_enabled": bool((cfg.get("finetune", {}) or {}).get("enabled", False)),
        },
        seed_dir / "EarlyFusion_MLP_params.json",
    )

    bp = study.best_params
    model = SimpleMLP(X_ef_tr.shape[1], bp["nl"], bp["nu"], bp["do"])
    model = train_torch(model, X_ef_tr, y_tr, device=device, lr=bp["lr"], epochs=15)

    out = {
        "train_proba": predict_torch(model, X_ef_tr, device=device),
        "test_proba": predict_torch(model, X_ef_te, device=device),
    }
    joblib.dump(out, ef_path)

    pr = average_precision_score(y_te, out["test_proba"])
    roc = roc_auc_score(y_te, out["test_proba"])
    log_print(f"[EarlyFusion] seed={seed} PR-AUC={pr:.4f} ROC-AUC={roc:.4f}")