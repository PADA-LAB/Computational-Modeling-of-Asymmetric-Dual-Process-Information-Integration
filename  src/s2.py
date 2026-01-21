# src/s2.py
from __future__ import annotations

import gc
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils.torch_utils import SimpleMLP, train_torch, predict_torch
from .utils import save_json, log_print, resolve_cfg_paths_abs, should_use_finetuned_t5

# T5 임베딩(LoRA / Base)
from .embedding.t5_lora_embedding import build_t5_lora_embeddings
from .embedding.t5_base_embedding import build_t5_base_embeddings

# 공통 캐시
from .embedding.cache_utils import embedding_cache_file


def load_or_make_full_embeddings(
    *,
    texts: list[str],
    cfg: dict,
    platform: str,
    device: torch.device,
    batch_size: int,
    platform_out_dir: Path,
) -> np.ndarray:
    """
    T5만 사용:
    - finetune.enabled=True  -> LoRA adapter 적용한 T5 encoder 임베딩
    - finetune.enabled=False -> base T5 encoder 임베딩
    """
    cache_path = embedding_cache_file(platform_out_dir, cfg)

    if cache_path.exists():
        X_all = np.load(cache_path, allow_pickle=False)
        if X_all.shape[0] != len(texts):
            raise ValueError(
                f"[S2] Embedding cache row mismatch.\n"
                f" - cache rows: {X_all.shape[0]}\n"
                f" - texts rows: {len(texts)}\n"
                f"cache: {cache_path}\n"
                f"다른 데이터로 생성된 캐시일 수 있어. 캐시 삭제 후 재실행."
            )
        return X_all

    # project_root 기준으로 paths 절대화
    project_root = Path(cfg["_project_root"]).resolve()
    paths_abs = resolve_cfg_paths_abs(cfg, project_root)

    # 캐시 저장 폴더 = cache_path.parent (cache_utils 규칙과 동일)
    out_dir = cache_path.parent

    emb = cfg.get("embedding", {}) or {}
    base_model_name = emb.get("base_model_name", "t5-base")
    max_length = emb.get("max_length", 256)
    pool = emb.get("pool", "mean")

    if should_use_finetuned_t5(cfg):
        lora_model_dir = Path(paths_abs["finetune_model_dir"]) / platform
        if not lora_model_dir.exists():
            raise FileNotFoundError(
                f"[S2] LoRA adapter dir not found: {lora_model_dir}\n"
                f"runner가 플랫폼 시작 시 finetune을 수행하도록 되어 있어야 합니다."
            )

        X_all = build_t5_lora_embeddings(
            texts=texts,
            base_model_name=base_model_name,
            lora_dir=lora_model_dir,
            out_dir=out_dir,
            max_length=max_length,
            batch_size=batch_size,
            pool=pool,
            device=device,
        )
    else:
        X_all = build_t5_base_embeddings(
            texts=texts,
            base_model_name=base_model_name,
            out_dir=out_dir,
            max_length=max_length,
            batch_size=batch_size,
            pool=pool,
            device=device,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    X_all = X_all.astype(np.float32)
    np.save(cache_path, X_all)
    log_print(f"[S2] saved embedding cache: {cache_path} shape={X_all.shape}")
    return X_all


def run_s2_train_and_save(
    *,
    cfg: dict,
    platform: str,
    df: pd.DataFrame,
    text_col: str,
    y_col: str,
    seed: int,
    s2_models: list[str],
    n_trials: int,
    seed_dir: Path,
    platform_out_dir: Path,
    batch_size: int,
):
    tr_idx = np.load(seed_dir / "train_idx.npy", allow_pickle=False)
    te_idx = np.load(seed_dir / "test_idx.npy", allow_pickle=False)

    texts = df[text_col].astype(str).tolist()
    y_all = df[y_col].values.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_all = load_or_make_full_embeddings(
        texts=texts,
        cfg=cfg,
        platform=platform,
        device=device,
        batch_size=batch_size,
        platform_out_dir=platform_out_dir,
    )

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

    cache_path_str = str(embedding_cache_file(platform_out_dir, cfg))
    finetune_enabled = bool((cfg.get("finetune", {}) or {}).get("enabled", False))
    embedding_mode = "t5"  # ✅ 고정

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

            save_json(
                {
                    "seed": seed,
                    "model": s2_name,
                    "best_score": study.best_value,
                    "best_params": study.best_params,
                    "embedding_cache_file": cache_path_str,
                    "embedding_mode": embedding_mode,
                    "finetune_enabled": finetune_enabled,
                },
                seed_dir / f"S2_only_{s2_name}_params.json",
            )

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

            save_json(
                {
                    "seed": seed,
                    "model": s2_name,
                    "best_score": study.best_value,
                    "best_params": study.best_params,
                    "embedding_cache_file": cache_path_str,
                    "embedding_mode": embedding_mode,
                    "finetune_enabled": finetune_enabled,
                },
                seed_dir / f"S2_only_{s2_name}_params.json",
            )

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