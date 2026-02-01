from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, precision_recall_curve
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)
    m: dict[str, float] = {}

    try:
        m["accuracy"] = float(accuracy_score(y_true, y_pred))
        m["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        m["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        m["accuracy"] = m["precision"] = m["recall"] = m["f1"] = float("nan")

    try:
        m["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        m["pr_auc"] = float("nan")

    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        m["roc_auc"] = float("nan")

    m["threshold"] = float(threshold)
    return m


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> tuple[float, float]:
    """
    PR curve 기반으로 F1 최대 threshold 선택
    returns: (best_threshold, best_f1)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, float("nan")

    f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])