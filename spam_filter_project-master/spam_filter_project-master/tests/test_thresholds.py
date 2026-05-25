from __future__ import annotations

import numpy as np

from src.evaluation.thresholds import select_threshold_by_precision_floor, threshold_sweep


def test_threshold_sweep_includes_expected_grid():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    table = threshold_sweep(y_true, y_prob, start=0.05, end=0.95, step=0.01)
    assert len(table) == 91
    assert round(float(table.iloc[0]["threshold"]), 2) == 0.05
    assert round(float(table.iloc[-1]["threshold"]), 2) == 0.95


def test_select_threshold_prefers_center_when_tied():
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    threshold, table = select_threshold_by_precision_floor(
        y_true,
        y_prob,
        precision_floor=0.95,
        start=0.05,
        end=0.95,
        step=0.01,
    )
    assert threshold == 0.5
    assert table["precision"].max() == 1.0
