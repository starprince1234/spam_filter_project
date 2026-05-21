from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_report_at_threshold


def threshold_sweep(y_true, y_prob, start: float = 0.05, end: float = 0.95, step: float = 0.01) -> pd.DataFrame:
    thresholds = np.round(np.arange(start, end + 1e-9, step), 2)
    rows = []
    for thr in thresholds:
        report = classification_report_at_threshold(y_true, y_prob, threshold=float(thr))
        rows.append(
            {
                "threshold": float(thr),
                "precision": report["precision"],
                "recall": report["recall"],
                "f1": report["f1"],
                "macro_f1": report["macro_f1"],
                "pr_auc": report["pr_auc"],
                "roc_auc": report["roc_auc"],
                "brier": report["brier"],
                "ece": report["ece"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold_by_precision_floor(
    y_true,
    y_prob,
    precision_floor: float = 0.95,
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
):
    table = threshold_sweep(y_true, y_prob, start=start, end=end, step=step)
    valid = table[table["precision"] >= precision_floor].copy()
    if valid.empty:
        best = table.sort_values(["precision", "f1", "recall", "threshold"], ascending=[False, False, False, True]).iloc[0]
        return float(best["threshold"]), table
    valid["distance_to_mid"] = (valid["threshold"] - 0.5).abs()
    best = valid.sort_values(["f1", "recall", "distance_to_mid"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), table
