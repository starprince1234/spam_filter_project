from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= 0.5).astype(int)
    conf = np.where(pred == 1, prob, 1.0 - prob)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= threshold).astype(int)
    out: dict[str, float | list[list[int]]] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": expected_calibration_error(y_true, prob),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def select_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    min_precision: float = 0.95,
    metric: str = "f1",
) -> tuple[float, dict[str, float | list[list[int]]]]:
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, prob, best_threshold)
    best_score = -1.0
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metrics = binary_metrics(y_true, prob, float(threshold))
        if float(metrics["precision"]) < min_precision:
            continue
        score = float(metrics.get(metric, metrics["f1"]))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_score < 0:
        return 0.5, best_metrics
    return best_threshold, best_metrics

