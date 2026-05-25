from __future__ import annotations

import numpy as np

from src.evaluation.thresholds import select_threshold_by_precision_floor


def test_threshold_selection_returns_float_and_table():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.95])
    threshold, table = select_threshold_by_precision_floor(y_true, y_prob, precision_floor=0.8)
    assert isinstance(threshold, float)
    assert not table.empty
