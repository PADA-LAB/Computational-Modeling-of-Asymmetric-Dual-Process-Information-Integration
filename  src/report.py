# src/report.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


def find_best_threshold_f1(y_true, y_prob):
    best_th, best_f1 = 0.5, -1
    for th in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (y_prob >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
    return best_th


def get_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Threshold": float(threshold),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Acc": float(accuracy_score(y_true, y_pred)),
        "Prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "Rec": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _load_json(path: Path):
    if not path.exists():
        return "N/A"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except:
        return "Error"


def build_report(platform_out_dir: Path, platform: str, seeds: list[int], s1_models: list[str]):
    rows = []

    for seed in seeds:
        seed_dir = platform_out_dir / f"seed_{seed}"
        if not (seed_dir / "done_s1.flag").exists():
            continue

        # GT
        y_tr = y_te = None
        for m in s1_models:
            tr_csv = seed_dir / f"S1_{m}" / "s1_train_preds.csv"
            te_csv = seed_dir / f"S1_{m}" / "s1_pred_proba.csv"
            if tr_csv.exists() and te_csv.exists():
                y_tr = pd.read_csv(tr_csv)["y_true"].values
                y_te = pd.read_csv(te_csv)["y_true"].values
                break
        if y_tr is None:
            continue

        # --- S1 ---
        for m in s1_models:
            tr_csv = seed_dir / f"S1_{m}" / "s1_train_preds.csv"
            te_csv = seed_dir / f"S1_{m}" / "s1_pred_proba.csv"
            p_tr = pd.read_csv(tr_csv)["s1_pred_proba"].values
            p_te = pd.read_csv(te_csv)["s1_pred_proba"].values
            th = find_best_threshold_f1(y_tr, p_tr)
            met = get_metrics(y_te, p_te, th)
            met.update({"seed": seed, "method": "S1_Only", "model": m, "Best_Params": _load_json(seed_dir / f"S1_{m}/best_params.json")})
            rows.append(met)

        # --- S2 ---
        s2_pkl = seed_dir / "s2_preds_dict.pkl"
        if s2_pkl.exists():
            s2_preds = joblib.load(s2_pkl)
            for m in s2_preds["test"].keys():
                p_tr = s2_preds["train"][m]
                p_te = s2_preds["test"][m]
                th = find_best_threshold_f1(y_tr, p_tr)
                met = get_metrics(y_te, p_te, th)
                met.update({"seed": seed, "method": "S2_Only", "model": m, "Best_Params": _load_json(seed_dir / f"S2_only_{m}_params.json")})
                rows.append(met)

        # --- Early Fusion ---
        ef_pkl = seed_dir / "early_fusion_preds.pkl"
        if ef_pkl.exists():
            ef = joblib.load(ef_pkl)
            th = find_best_threshold_f1(y_tr, ef["train_proba"])
            met = get_metrics(y_te, ef["test_proba"], th)
            met.update({"seed": seed, "method": "Early_Fusion", "model": "MLP", "Best_Params": _load_json(seed_dir / "EarlyFusion_MLP_params.json")})
            rows.append(met)

        # --- Residual Forward/Reverse ---
        fwd_pkl = seed_dir / "residual_forward_preds.pkl"
        if fwd_pkl.exists():
            fwd = joblib.load(fwd_pkl)
            # base S1 preds 로드
            s1_test_map = {m: pd.read_csv(seed_dir / f"S1_{m}/s1_pred_proba.csv")["s1_pred_proba"].values for m in s1_models}
            s1_train_map = {m: pd.read_csv(seed_dir / f"S1_{m}/s1_train_preds.csv")["s1_pred_proba"].values for m in s1_models}

            for s1_base, sub in fwd.items():
                for s2_reg, res in sub.items():
                    p_base_te = s1_test_map[s1_base]
                    p_base_tr = s1_train_map[s1_base]
                    pf_te = np.clip(p_base_te + res["test_resid"], 0, 1)
                    pf_tr = np.clip(p_base_tr + res["train_resid"], 0, 1)
                    th = find_best_threshold_f1(y_tr, pf_tr)
                    met = get_metrics(y_te, pf_te, th)
                    met.update({"seed": seed, "method": "Residual_Forward", "model": f"{s1_base}->{s2_reg}", "Best_Params": _load_json(seed_dir / f"Residual_Fwd_{s1_base}_to_{s2_reg}_params.json")})
                    rows.append(met)

        rev_pkl = seed_dir / "residual_reverse_preds.pkl"
        if rev_pkl.exists() and s2_pkl.exists():
            rev = joblib.load(rev_pkl)
            s2_preds = joblib.load(s2_pkl)
            for s2_base, regs in rev.items():
                for s1_reg, res in regs.items():
                    p_base_te = s2_preds["test"][s2_base]
                    p_base_tr = s2_preds["train"][s2_base]
                    pf_te = np.clip(p_base_te + res["test_resid"], 0, 1)
                    pf_tr = np.clip(p_base_tr + res["train_resid"], 0, 1)
                    th = find_best_threshold_f1(y_tr, pf_tr)
                    met = get_metrics(y_te, pf_te, th)
                    met.update({"seed": seed, "method": "Residual_Reverse", "model": f"{s2_base}->{s1_reg}", "Best_Params": _load_json(seed_dir / f"Residual_Rev_{s2_base}_to_{s1_reg}_params.json")})
                    rows.append(met)

    df = pd.DataFrame(rows)
    report_dir = platform_out_dir / "final_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(report_dir / f"{platform}_final_report_raw_seeds.csv", index=False)

    metric_cols = ["AUPRC", "ROC_AUC", "F1", "Acc", "Prec", "Rec"]
    if df.empty:
        return

    summary = df.groupby(["method", "model"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in summary.columns]
    summary.to_csv(report_dir / f"{platform}_final_report_summary_stats.csv", index=False)