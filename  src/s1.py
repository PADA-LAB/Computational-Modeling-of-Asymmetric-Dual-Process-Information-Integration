# src/s1.py
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import ensure_dir, save_json, log_print


def run_s1_train_and_save(
    df: pd.DataFrame,
    s1_feature_cols: list[str],
    y_col: str,
    seed: int,
    s1_models: list[str],
    n_trials: int,
    seed_dir: Path,
):
    ensure_dir(seed_dir)

    y_raw = df[y_col].values.astype(int)

    # early_fusion과 동일하게 robust하게 처리
    use_cols = [c for c in s1_feature_cols if c in df.columns]
    X_raw = df[use_cols].apply(pd.to_numeric, errors="coerce").values
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    indices = np.arange(len(df))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y_raw)
    np.save(seed_dir / "train_idx.npy", tr_idx)
    np.save(seed_dir / "test_idx.npy", te_idx)

    X_train, X_test = X_raw[tr_idx], X_raw[te_idx]
    y_train, y_test = y_raw[tr_idx], y_raw[te_idx]

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    for m_name in s1_models:
        save_path = ensure_dir(seed_dir / f"S1_{m_name}")

        def obj(trial):
            if m_name == "lgbm":
                p = {
                    "n_estimators": trial.suggest_int("n", 100, 300),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.1),
                    "verbose": -1,
                    "random_state": seed,
                }
                model = lgb.LGBMClassifier(**p)
            elif m_name == "xgb":
                p = {
                    "n_estimators": trial.suggest_int("n", 100, 300),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.1),
                    "verbosity": 0,
                    "random_state": seed,
                }
                model = xgb.XGBClassifier(**p)
            else:
                p = {
                    "iterations": trial.suggest_int("n", 100, 300),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.1),
                    "verbose": False,
                    "random_seed": seed,
                }
                model = cat.CatBoostClassifier(**p)

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            scores = []
            for t, v in skf.split(X_train, y_train):
                model.fit(X_train[t], y_train[t])
                p = model.predict_proba(X_train[v])[:, 1]
                scores.append(average_precision_score(y_train[v], p))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=n_trials)

        bp = study.best_params
        save_json(bp, save_path / "best_params.json")

        if m_name == "lgbm":
            clf = lgb.LGBMClassifier(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbose=-1)
        elif m_name == "xgb":
            clf = xgb.XGBClassifier(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbosity=0)
        else:
            clf = cat.CatBoostClassifier(iterations=bp["n"], learning_rate=bp["lr"], random_seed=seed, verbose=False)

        clf.fit(X_train, y_train)

        train_proba = clf.predict_proba(X_train)[:, 1]
        test_proba = clf.predict_proba(X_test)[:, 1]

        pd.DataFrame({"s1_pred_proba": train_proba, "y_true": y_train}).to_csv(save_path / "s1_train_preds.csv", index=False)
        pd.DataFrame({"s1_pred_proba": test_proba, "y_true": y_test}).to_csv(save_path / "s1_pred_proba.csv", index=False)

        pr_val = average_precision_score(y_test, test_proba)
        roc_val = roc_auc_score(y_test, test_proba)
        log_print(f"[S1] seed={seed} model={m_name} PR-AUC={pr_val:.4f} ROC-AUC={roc_val:.4f}")

    (seed_dir / "done_s1.flag").write_text("done", encoding="utf-8")