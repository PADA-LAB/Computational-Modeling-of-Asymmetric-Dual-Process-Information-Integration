# review_helpfulness_runner/early_fusion.py
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

from .torch_utils import SimpleMLP, train_torch, predict_torch
from .utils import save_json, log_print

from src.embedding.embedding_utils import get_embeddings


def run_early_fusion(
    df: pd.DataFrame,
    s1_feature_cols: list[str],
    text_col: str,
    y_col: str,
    seed: int,
    seed_dir: Path,
    platform_out_dir: Path,
    embedding_model: str,
    batch_size: int,
    n_trials: int,
):
    ef_path = seed_dir / "early_fusion_preds.pkl"
    if ef_path.exists():
        return

    tr_idx = np.load(seed_dir / "train_idx.npy")
    te_idx = np.load(seed_dir / "test_idx.npy")

    y_all = df[y_col].values.astype(int)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    X_s1 = df[[c for c in s1_feature_cols if c in df.columns]].values
    X_s1_tr, X_s1_te = X_s1[tr_idx], X_s1[te_idx]

    texts = df[text_col].astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 임베딩은 cache를 s2.py에서 만들었을 수도 있지만, 여기서는 단순화를 위해 재생성/캐시 재사용을 원하면 s2.py 방식으로 합쳐도 됨
    # (속도 중요하면 s2.py의 cache_dir/X_all.npy를 읽어오는 방식으로 바꾸면 됨)
    X_s2_all = get_embeddings(texts, embedding_model, device=device, batch_size=batch_size)
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
        {"seed": seed, "model": "EarlyFusion_MLP", "best_score": study.best_value, "best_params": study.best_params,
         "embedding_model": embedding_model},
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