# src/residual.py
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import joblib
import torch
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler

from .utils import SimpleMLP, train_torch, predict_torch, save_json, log_print
from .embedding.cache_utils import embedding_cache_file


def _load_s2_cached_embeddings(*, platform_out_dir: Path, cfg: dict, expected_rows: int) -> np.ndarray:
    cache_path = embedding_cache_file(platform_out_dir, cfg)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"[Residual] S2 embedding cache not found: {cache_path}\n"
            f"S2를 먼저 실행해서 X_all.npy를 생성해야 합니다."
        )

    X = np.load(cache_path, allow_pickle=False)
    if X.shape[0] != expected_rows:
        raise ValueError(
            f"[Residual] Embedding cache row mismatch:\n"
            f" - cache rows: {X.shape[0]}\n"
            f" - df rows   : {expected_rows}\n"
            f"같은 df / 같은 순서로 만든 캐시인지 확인해줘."
        )
    return X


def run_residual_both_directions(
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
    s1_models: list[str],
    resid_s2_models: list[str],
):
    tr_idx = np.load(seed_dir / "train_idx.npy")
    te_idx = np.load(seed_dir / "test_idx.npy")

    y_all = df[y_col].values.astype(int)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    # early_fusion처럼 robust numeric 처리
    use_cols = [c for c in s1_feature_cols if c in df.columns]
    X_s1_all = df[use_cols].apply(pd.to_numeric, errors="coerce").values
    X_s1_all = np.nan_to_num(X_s1_all, nan=0.0, posinf=0.0, neginf=0.0)

    X_s1_tr = X_s1_all[tr_idx]
    X_s1_te = X_s1_all[te_idx]

    scaler = RobustScaler()
    X_s1_tr = scaler.fit_transform(X_s1_tr)
    X_s1_te = scaler.transform(X_s1_te)

    X_s2_all = _load_s2_cached_embeddings(platform_out_dir=platform_out_dir, cfg=cfg, expected_rows=len(df))
    X_s2_tr, X_s2_te = X_s2_all[tr_idx], X_s2_all[te_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Forward Residual ----------
    fwd_path = seed_dir / "residual_forward_preds.pkl"
    fwd_preds = joblib.load(fwd_path) if fwd_path.exists() else {}

    for s1_name in s1_models:
        s1_tr_csv = seed_dir / f"S1_{s1_name}" / "s1_train_preds.csv"
        s1_te_csv = seed_dir / f"S1_{s1_name}" / "s1_pred_proba.csv"
        if not s1_tr_csv.exists():
            continue

        p_s1_tr = pd.read_csv(s1_tr_csv)["s1_pred_proba"].values
        p_s1_te = pd.read_csv(s1_te_csv)["s1_pred_proba"].values
        y_resid = y_tr - p_s1_tr

        fwd_preds.setdefault(s1_name, {})
        for s2_reg in resid_s2_models:
            if s2_reg in fwd_preds[s1_name]:
                continue

            optuna.logging.set_verbosity(optuna.logging.ERROR)

            if s2_reg == "mlp":
                def obj(trial):
                    nl = trial.suggest_int("nl", 1, 3)
                    nu = trial.suggest_int("nu", 64, 256)
                    do = trial.suggest_float("do", 0.1, 0.5)
                    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
                    maes = []
                    for t, v in kf.split(X_s2_tr):
                        m = SimpleMLP(X_s2_tr.shape[1], nl, nu, do, output_dim=1, task_type="regression")
                        m = train_torch(m, X_s2_tr[t], y_resid[t], device=device, lr=lr, epochs=5, task_type="regression")
                        p = predict_torch(m, X_s2_tr[v], device=device)
                        maes.append(mean_absolute_error(y_resid[v], p))
                    return float(np.mean(maes))

                study = optuna.create_study(direction="minimize")
                study.optimize(obj, n_trials=n_trials)

                save_json(
                    {"seed": seed, "model": f"Residual_Fwd_{s1_name}_to_{s2_reg}",
                     "best_score": study.best_value, "best_params": study.best_params,
                     "embedding_key": cfg.get("embedding", {})},
                    seed_dir / f"Residual_Fwd_{s1_name}_to_{s2_reg}_params.json",
                )

                bp = study.best_params
                final = SimpleMLP(X_s2_tr.shape[1], bp["nl"], bp["nu"], bp["do"], output_dim=1, task_type="regression")
                final = train_torch(final, X_s2_tr, y_resid, device=device, lr=bp["lr"], epochs=15, task_type="regression")
                res_tr = predict_torch(final, X_s2_tr, device=device)
                res_te = predict_torch(final, X_s2_te, device=device)

            else:
                def obj_tree(trial):
                    n_est = trial.suggest_int("n", 100, 300)
                    lr = trial.suggest_float("lr", 0.01, 0.1)
                    if s2_reg == "lgbm":
                        m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, verbose=-1, random_state=seed)
                    elif s2_reg == "xgb":
                        m = xgb.XGBRegressor(n_estimators=n_est, learning_rate=lr, verbosity=0, random_state=seed)
                    else:
                        m = cat.CatBoostRegressor(iterations=n_est, learning_rate=lr, verbose=False, random_seed=seed)

                    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
                    maes = []
                    for t, v in kf.split(X_s2_tr):
                        m.fit(X_s2_tr[t], y_resid[t])
                        p = m.predict(X_s2_tr[v])
                        maes.append(mean_absolute_error(y_resid[v], p))
                    return float(np.mean(maes))

                study = optuna.create_study(direction="minimize")
                study.optimize(obj_tree, n_trials=n_trials)

                save_json(
                    {"seed": seed, "model": f"Residual_Fwd_{s1_name}_to_{s2_reg}",
                     "best_score": study.best_value, "best_params": study.best_params,
                     "embedding_key": cfg.get("embedding", {})},
                    seed_dir / f"Residual_Fwd_{s1_name}_to_{s2_reg}_params.json",
                )

                bp = study.best_params
                if s2_reg == "lgbm":
                    reg = lgb.LGBMRegressor(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbose=-1)
                elif s2_reg == "xgb":
                    reg = xgb.XGBRegressor(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbosity=0)
                else:
                    reg = cat.CatBoostRegressor(iterations=bp["n"], learning_rate=bp["lr"], random_seed=seed, verbose=False)

                reg.fit(X_s2_tr, y_resid)
                res_tr = reg.predict(X_s2_tr)
                res_te = reg.predict(X_s2_te)

            fwd_preds[s1_name][s2_reg] = {"train_resid": res_tr, "test_resid": res_te}
            joblib.dump(fwd_preds, fwd_path)

            final_prob = np.clip(p_s1_te + res_te, 0, 1)
            pr = average_precision_score(y_te, final_prob)
            log_print(f"[FwdResid] seed={seed} {s1_name}->{s2_reg} PR-AUC={pr:.4f}")

    # ---------- Reverse Residual ----------
    rev_path = seed_dir / "residual_reverse_preds.pkl"
    s2_preds_path = seed_dir / "s2_preds_dict.pkl"
    if not s2_preds_path.exists():
        return

    s2_preds_data = joblib.load(s2_preds_path)
    rev_results = joblib.load(rev_path) if rev_path.exists() else {}

    for s2_name in list(s2_preds_data["train"].keys()):
        p_s2_tr = s2_preds_data["train"][s2_name]
        p_s2_te = s2_preds_data["test"][s2_name]
        y_resid = y_tr - p_s2_tr
        rev_results.setdefault(s2_name, {})

        for s1_reg in ["lgbm", "xgb", "cat"]:
            if s1_reg in rev_results[s2_name]:
                continue

            def obj_rev(trial):
                n_est = trial.suggest_int("n", 100, 300)
                lr = trial.suggest_float("lr", 0.01, 0.1)
                if s1_reg == "lgbm":
                    m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, verbose=-1, random_state=seed)
                elif s1_reg == "xgb":
                    m = xgb.XGBRegressor(n_estimators=n_est, learning_rate=lr, verbosity=0, random_state=seed)
                else:
                    m = cat.CatBoostRegressor(iterations=n_est, learning_rate=lr, verbose=False, random_seed=seed)

                kf = KFold(n_splits=3, shuffle=True, random_state=seed)
                maes = []
                for t, v in kf.split(X_s1_tr):
                    m.fit(X_s1_tr[t], y_resid[t])
                    p = m.predict(X_s1_tr[v])
                    maes.append(mean_absolute_error(y_resid[v], p))
                return float(np.mean(maes))

            study = optuna.create_study(direction="minimize")
            study.optimize(obj_rev, n_trials=n_trials)

            save_json(
                {"seed": seed, "model": f"Residual_Rev_{s2_name}_to_{s1_reg}",
                 "best_score": study.best_value, "best_params": study.best_params,
                 "embedding_key": cfg.get("embedding", {})},
                seed_dir / f"Residual_Rev_{s2_name}_to_{s1_reg}_params.json",
            )

            bp = study.best_params
            if s1_reg == "lgbm":
                reg = lgb.LGBMRegressor(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbose=-1)
            elif s1_reg == "xgb":
                reg = xgb.XGBRegressor(n_estimators=bp["n"], learning_rate=bp["lr"], random_state=seed, verbosity=0)
            else:
                reg = cat.CatBoostRegressor(iterations=bp["n"], learning_rate=bp["lr"], random_seed=seed, verbose=False)

            reg.fit(X_s1_tr, y_resid)
            res_tr = reg.predict(X_s1_tr)
            res_te = reg.predict(X_s1_te)

            rev_results[s2_name][s1_reg] = {"train_resid": res_tr, "test_resid": res_te}
            joblib.dump(rev_results, rev_path)

            final_prob = np.clip(p_s2_te + res_te, 0, 1)
            pr = average_precision_score(y_te, final_prob)
            log_print(f"[RevResid] seed={seed} {s2_name}->{s1_reg} PR-AUC={pr:.4f}")