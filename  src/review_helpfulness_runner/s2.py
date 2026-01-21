# review_helpfulness_runner/s2.py
from __future__ import annotations

import gc
import numpy as np
import pandas as pd
import optuna
import joblib
import torch
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from .torch_utils import SimpleMLP, train_torch, predict_torch
from .utils import ensure_dir, save_json, log_print

# ✅ 네 embedding_utils.py 그대로 가져와 쓰는 방식
# (runner 폴더 안에 embedding_utils를 만들지 않으려면, import 경로만 맞춰주면 됨)
# repo 구조가 repo/src/embedding/embedding_utils.py 라고 했으니:
from src.embedding.embedding_utils import get_embeddings


def _embedding_cache_path(out_dir: Path, embedding_model: str) -> Path:
    safe_name = embedding_model.replace("/", "__")
    return ensure_dir(out_dir / "embeddings_cache" / safe_name)


def load_or_make_full_embeddings(
    texts: list[str],
    embedding_model: str,
    device: torch.device,
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    """
    full embeddings를 1번 만들고 cache_dir/X_all.npy 로 저장해서 재사용
    """
    cache_path = cache_dir / "X_all.npy"
    if cache_path.exists():
        return np.load(cache_path)

    X_all = get_embeddings(texts, embedding_model, device=device, batch_size=batch_size)
    np.save(cache_path, X_all)
    return X_all


def run_s2_train_and_save(
    df: pd.DataFrame,
    text_col: str,
    y_col: str,
    seed: int,
    s2_models: list[str],
    n_trials: int,
    seed_dir: Path,
    platform_out_dir: Path,
    embedding_model: str,
    batch_size: int,
):
    """
    - seed_dir/train_idx.npy, test_idx.npy 를 이용해 slicing
    - seed_dir/s2_preds_dict.pkl 저장
    - seed_dir/S2_only_{model}_params.json 저장
    """
    tr_idx = np.load(seed_dir / "train_idx.npy")
    te_idx = np.load(seed_dir / "test_idx.npy")

    texts = df[text_col].astype(str).tolist()
    y_all = df[y_col].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = _embedding_cache_path(platform_out_dir, embedding_model)
    X_all = load_or_make_full_embeddings(texts, embedding_model, device, batch_size, cache_dir)

    X_tr = X_all[tr_idx]
    X_te = X_all[te_idx]
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    s2_res_path = seed_dir / "s2_preds_dict.pkl"
    if s2_res_path.exists():
        s2_preds = joblib.load(s2_res_path)
    else:
        s2_preds = {"train": {}, "test": {}}

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    for s2_name in s2_models:
        if s2_name in s2_preds["test"]:
            continue

        if s2_name == "mlp":
            def obj_mlp(trial):
                nl = trial.suggest_int("nl", 1, 3)
                nu = trial.suggest_int("nu", 64, 256)
                do = trial.suggest_float("do", 0.1, 0.5)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                scores = []
                for t, v in skf.split(X_tr, y_tr):
                    m = SimpleMLP(X_tr.shape[1], nl, nu, do)
                    m = train_torch(m, X_tr[t], y_tr[t], device=device, lr=lr, epochs=5)
                    p = predict_torch(m, X_tr[v], device=device)
                    scores.append(average_precision_score(y_tr[v], p))
                return float(np.mean(scores))

            study = optuna.create_study(direction="maximize")
            study.optimize(obj_mlp, n_trials=n_trials)

            best_info = {
                "seed": seed, "model": s2_name,
                "best_score": study.best_value, "best_params": study.best_params,
                "embedding_model": embedding_model,
            }
            save_json(best_info, seed_dir / f"S2_only_{s2_name}_params.json")

            bp = study.best_params
            final_model = SimpleMLP(X_tr.shape[1], bp["nl"], bp["nu"], bp["do"])
            final_model = train_torch(final_model, X_tr, y_tr, device=device, lr=bp["lr"], epochs=15)

            s2_preds["train"][s2_name] = predict_torch(final_model, X_tr, device=device)
            s2_preds["test"][s2_name] = predict_torch(final_model, X_te, device=device)

        else:
            def obj_tree(trial):
                n_est = trial.suggest_int("n", 100, 300)
                lr = trial.suggest_float("lr", 0.01, 0.1)

                if s2_name == "lgbm":
                    m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=lr, verbose=-1, random_state=seed)
                elif s2_name == "xgb":
                    m = xgb.XGBClassifier(n_estimators=n_est, learning_rate=lr, verbosity=0, random_state=seed)
                else:
                    m = cat.CatBoostClassifier(iterations=n_est, learning_rate=lr, verbose=False, random_seed=seed)

                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                scores = []
                for t, v in skf.split(X_tr, y_tr):
                    m.fit(X_tr[t], y_tr[t])
                    p = m.predict_proba(X_tr[v])[:, 1]
                    scores.append(average_precision_score(y_tr[v], p))
                return float(np.mean(scores))

            study = optuna.create_study(direction="maximize")
            study.optimize(obj_tree, n_trials=n_trials)

            best_info = {
                "seed": seed, "model": s2_name,
                "best_score": study.best_value, "best_params": study.best_params,
                "embedding_model": embedding_model,
            }
            save_json(best_info, seed_dir / f"S2_only_{s2_name}_params.json")

            bp = study.best_params
            if s2_name == "lgbm":
                clf = lgb.LGBMClassifier(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbose=-1)
            elif s2_name == "xgb":
                clf = xgb.XGBClassifier(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbosity=0)
            else:
                clf = cat.CatBoostClassifier(iterations=bp["n"], learning_rate=bp["lr"], random_seed=seed, verbose=False)

            clf.fit(X_tr, y_tr)
            s2_preds["train"][s2_name] = clf.predict_proba(X_tr)[:, 1]
            s2_preds["test"][s2_name] = clf.predict_proba(X_te)[:, 1]

        joblib.dump(s2_preds, s2_res_path)

        pr_val = average_precision_score(y_te, s2_preds["test"][s2_name])
        roc_val = roc_auc_score(y_te, s2_preds["test"][s2_name])
        log_print(f"[S2] seed={seed} model={s2_name} PR-AUC={pr_val:.4f} ROC-AUC={roc_val:.4f}")

    del X_all, X_tr, X_te
    gc.collect()