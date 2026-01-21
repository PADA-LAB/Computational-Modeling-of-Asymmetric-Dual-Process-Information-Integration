# src/embedding/run_embedding_comparison.py
import gc
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from .embedding_utils import get_embeddings
from .mlp_utils import SimpleMLP, train_torch, predict_torch
from .paths import resolve_data_path, make_output_path

def run_embedding_pipeline(cfg: dict, platform: str, project_root):
    exp = cfg["experiment"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    if platform not in data_cfg:
        raise ValueError(f"Unknown platform: {platform}. Available: {list(data_cfg.keys())}")

    data_path = resolve_data_path(project_root, data_cfg[platform])
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    out_dir = make_output_path(project_root, out_cfg["root_dir"], platform)
    out_csv = out_dir / f"{platform}_S2_Embedding_Comparison_Final.csv"

    text_col = exp["text_column"]
    label_col = exp["label_column"]
    seeds = exp["seeds"]
    n_trials = exp["n_trials"]
    embedding_models = exp["embedding_models"]
    s2_models = exp["s2_models"]
    batch_size = exp.get("batch_size", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].values.astype(int)

    results = []

    for emb_name in embedding_models:
        X_all = get_embeddings(texts, emb_name, device=device, batch_size=batch_size)

        for seed in seeds:
            indices = np.arange(len(labels))
            tr_idx, te_idx = train_test_split(
                indices, test_size=0.2, random_state=seed, stratify=labels
            )
            X_train, X_test = X_all[tr_idx], X_all[te_idx]
            y_train, y_test = labels[tr_idx], labels[te_idx]

            for s2_name in s2_models:
                print(f"👉 [{platform}] Emb:{emb_name.split('/')[-1]} | Seed:{seed} | Model:{s2_name}")

                def objective(trial):
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                    scores = []

                    if s2_name == "mlp":
                        nl = trial.suggest_int("nl", 1, 3)
                        nu = trial.suggest_int("nu", 64, 256)
                        do = trial.suggest_float("do", 0.1, 0.5)
                        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

                        for t, v in skf.split(X_train, y_train):
                            m = SimpleMLP(X_train.shape[1], nl, nu, do)
                            m, _ = train_torch(
                                m, X_train[t], y_train[t], X_train[v], y_train[v],
                                device=device, lr=lr, epochs=5
                            )
                            p = predict_torch(m, X_train[v], device=device)
                            scores.append(average_precision_score(y_train[v], p))
                    else:
                        n_est = trial.suggest_int("n", 100, 300)
                        lr = trial.suggest_float("lr", 0.01, 0.1)

                        if s2_name == "lgbm":
                            clf = lgb.LGBMClassifier(
                                n_estimators=n_est, learning_rate=lr, verbose=-1, random_state=seed
                            )
                        elif s2_name == "xgb":
                            clf = xgb.XGBClassifier(
                                n_estimators=n_est, learning_rate=lr, verbosity=0, random_state=seed
                            )
                        else:  # cat
                            clf = cat.CatBoostClassifier(
                                iterations=n_est, learning_rate=lr, verbose=False, random_seed=seed
                            )

                        for t, v in skf.split(X_train, y_train):
                            clf.fit(X_train[t], y_train[t])
                            p = clf.predict_proba(X_train[v])[:, 1]
                            scores.append(average_precision_score(y_train[v], p))

                    return float(np.mean(scores))

                optuna.logging.set_verbosity(optuna.logging.ERROR)
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials)
                bp = study.best_params

                if s2_name == "mlp":
                    Xt, Xv, yt, yv = train_test_split(
                        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
                    )
                    final_model = SimpleMLP(X_train.shape[1], bp["nl"], bp["nu"], bp["do"])
                    final_model, _ = train_torch(
                        final_model, Xt, yt, Xv, yv, device=device, lr=bp["lr"], epochs=15
                    )
                    final_preds = predict_torch(final_model, X_test, device=device)
                else:
                    if s2_name == "lgbm":
                        clf = lgb.LGBMClassifier(
                            n_estimators=bp["n"], learning_rate=bp["lr"], verbose=-1, random_state=seed
                        )
                    elif s2_name == "xgb":
                        clf = xgb.XGBClassifier(
                            n_estimators=bp["n"], learning_rate=bp["lr"], verbosity=0, random_state=seed
                        )
                    else:
                        clf = cat.CatBoostClassifier(
                            iterations=bp["n"], learning_rate=bp["lr"], verbose=False, random_seed=seed
                        )

                    clf.fit(X_train, y_train)
                    final_preds = clf.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, final_preds)
                pr = average_precision_score(y_test, final_preds)
                pred_labels = (final_preds >= 0.5).astype(int)
                acc = accuracy_score(y_test, pred_labels)
                f1 = f1_score(y_test, pred_labels)

                results.append({
                    "Platform": platform,
                    "Embedding": emb_name.split("/")[-1],
                    "Seed": seed,
                    "S2_Model": s2_name,
                    "ROC_AUC": auc,
                    "PR_AUC": pr,
                    "Accuracy": acc,
                    "F1_Score": f1,
                    "Best_Params": str(bp),
                })
                print(f"   -> Result: PR-AUC={pr:.4f}, ROC-AUC={auc:.4f}")

        # memory cleanup
        del X_all, X_train, X_test
        gc.collect()

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_csv, index=False)

    print("\n Final Summary:")
    print(res_df.groupby(["Embedding", "S2_Model"])[["PR_AUC", "ROC_AUC", "Accuracy", "F1_Score"]].mean())
    print(f"\n Saved: {out_csv}")