from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    labels = np.asarray(y_true).astype(int)
    probs = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        mask = (probs >= bins[index]) & (probs < bins[index + 1])
        if not np.any(mask):
            continue
        ece += abs(labels[mask].mean() - probs[mask].mean()) * mask.mean()
    return float(ece)


def classification_report_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(y_true).astype(int)
    probs = np.asarray(y_prob).astype(float)
    pred = (probs >= threshold).astype(int)
    return {
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": expected_calibration_error(labels, probs),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, start: float = 0.05, end: float = 0.95, step: float = 0.01) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(start, end + 1e-9, step), 2):
        report = classification_report_at_threshold(y_true, y_prob, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **{key: report[key] for key in ["precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc", "brier", "ece"]}})
    return pd.DataFrame(rows)


def select_threshold_by_precision_floor(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_floor: float = 0.95,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    table = threshold_sweep(y_true, y_prob, start=start, end=end, step=step)
    valid = table[table["precision"] >= precision_floor].copy()
    if valid.empty:
        best = table.sort_values(["precision", "f1", "recall", "threshold"], ascending=[False, False, False, True]).iloc[0]
        return float(best["threshold"]), table
    valid["distance_to_mid"] = (valid["threshold"] - 0.5).abs()
    best = valid.sort_values(["f1", "recall", "distance_to_mid"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), table


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    labels = np.asarray(y_true).astype(float)
    best_temperature = 1.0
    best_loss = float("inf")
    for candidate in np.linspace(0.5, 5.0, 91):
        probs = np.clip(sigmoid_np(logits / candidate), 1e-6, 1.0 - 1e-6)
        loss = float(np.mean(-(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(candidate)
    return best_temperature
