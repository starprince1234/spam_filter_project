from __future__ import annotations

import numpy as np

from src.calibration.calibration_utils import apply_temperature_scaling, fit_sigmoid_calibrator


def test_sigmoid_calibrator_returns_probs():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    calibrated = fit_sigmoid_calibrator(y_true, y_prob)
    out = calibrated.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    assert out.shape == y_prob.shape
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_temperature_scaling_keeps_range():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    out, temperature = apply_temperature_scaling(y_true, y_prob)
    assert out.shape == y_prob.shape
    assert temperature > 0
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
