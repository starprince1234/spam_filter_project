from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def fit_sigmoid_calibrator(y_true, y_prob):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel().clip(1e-6, 1 - 1e-6)
    pseudo = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(pseudo, y_true)
    return calibrator


def calibrate_probabilities_sigmoid(y_true, y_prob):
    calibrator = fit_sigmoid_calibrator(y_true, y_prob)
    return calibrator.predict_proba(np.log(np.asarray(y_prob).ravel().clip(1e-6, 1 - 1e-6) / (1 - np.asarray(y_prob).ravel().clip(1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1], calibrator


def apply_temperature_scaling(y_true, y_prob):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel().clip(1e-6, 1 - 1e-6)
    logits = np.log(y_prob / (1 - y_prob))
    temp = 1.0
    best_temp = 1.0
    best_loss = float("inf")
    for candidate in np.linspace(0.5, 5.0, 91):
        scaled = 1 / (1 + np.exp(-logits / candidate))
        loss = np.mean(-(y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled)))
        if loss < best_loss:
            best_loss = loss
            best_temp = float(candidate)
    out = 1 / (1 + np.exp(-logits / best_temp))
    return out, best_temp


def calibrate_classifier_cv(estimator, X, y, method: str = "sigmoid"):
    clf = CalibratedClassifierCV(estimator, method=method, cv=3)
    clf.fit(X, y)
    return clf
