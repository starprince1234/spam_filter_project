from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from src.utils.io_utils import ensure_dir

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / 'outputs' / 'figures'
RESULTS_DIR = ROOT / 'experiments' / 'results'


def _savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def _load_test_preds() -> tuple[pd.Series, pd.Series, pd.Series]:
    pred_path = RESULTS_DIR / 'logistic_regression_calibrated_valid_predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    valid = pd.read_csv(ROOT / 'data' / 'splits' / 'valid.csv')
    pred = pd.read_csv(pred_path)
    y_true = pred['y_true']
    y_prob = pred['prob_after']
    return y_true, y_prob, valid['label']


def generate_figures() -> None:
    valid = pd.read_csv(ROOT / 'data' / 'splits' / 'valid.csv')
    test = pd.read_csv(ROOT / 'data' / 'splits' / 'test.csv')
    cal_valid = pd.read_csv(RESULTS_DIR / 'logistic_regression_calibrated_valid_predictions.csv')
    model_summary = pd.read_csv(RESULTS_DIR / 'step5_model_summary.csv')

    # Confusion matrix from selected LR threshold on validation
    threshold = float(pd.read_csv(RESULTS_DIR / 'step8_threshold_analysis.csv').query("model == 'logistic_regression'")['selected_threshold'].iloc[0])
    y_true = test['label'].to_numpy()
    y_prob = cal_valid['prob_after'].to_numpy()
    if len(y_prob) != len(test):
        # fall back to validating curve on validation split when test predictions are unavailable
        y_prob = cal_valid['prob_after'].to_numpy()
        y_true = cal_valid['y_true'].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['ham', 'spam']).plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix')
    _savefig(FIG_DIR / 'confusion_matrix.png')

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title('ROC Curve')
    _savefig(FIG_DIR / 'roc_curve.png')

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title('Precision-Recall Curve')
    _savefig(FIG_DIR / 'pr_curve.png')

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
    ax.plot(prob_pred, prob_true, marker='o', label='LR calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    _savefig(FIG_DIR / 'calibration_curve.png')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    generate_figures()


if __name__ == '__main__':
    main()
