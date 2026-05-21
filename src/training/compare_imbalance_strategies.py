from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.evaluation.metrics import classification_report_at_threshold
from src.evaluation.thresholds import select_threshold_by_precision_floor
from src.utils.io_utils import ensure_dir, load_yaml, write_csv, write_json


def _load_splits(root: Path):
    train = pd.read_csv(root / "data/splits/train.csv")
    valid = pd.read_csv(root / "data/splits/valid.csv")
    test = pd.read_csv(root / "data/splits/test.csv")
    return train, valid, test


def _fit_vectorizer(train_texts):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=30000)
    X_train = vec.fit_transform(train_texts)
    return vec, X_train


def _train_lr(train, valid, class_weight=None):
    vec, X_train = _fit_vectorizer(train["clean_text"].fillna(""))
    X_valid = vec.transform(valid["clean_text"].fillna(""))
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, n_jobs=1)
    clf.fit(X_train, train["label"])
    valid_prob = clf.predict_proba(X_valid)[:, 1]
    return clf, vec, valid_prob


def _train_svm(train, valid, class_weight=None):
    vec, X_train = _fit_vectorizer(train["clean_text"].fillna(""))
    X_valid = vec.transform(valid["clean_text"].fillna(""))
    base = LinearSVC(class_weight=class_weight)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train, train["label"])
    valid_prob = clf.predict_proba(X_valid)[:, 1]
    return clf, vec, valid_prob


def compare_strategies(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    train, valid, _ = _load_splits(root)
    results_dir = root / cfg["paths"]["results"]
    logs_dir = root / cfg["paths"]["logs"]
    ensure_dir(results_dir)
    ensure_dir(logs_dir)

    rows = []
    strategy_specs = [
        ("none", None, 0.5),
        ("class_weight", "balanced", 0.5),
        ("threshold_adjust", "balanced", None),
        ("cost_sensitive", "balanced", None),
    ]

    for strategy_name, class_weight, default_threshold in strategy_specs:
        lr_model, _, lr_prob = _train_lr(train, valid, class_weight=class_weight)
        svm_model, _, svm_prob = _train_svm(train, valid, class_weight=class_weight)

        for model_name, prob in [("logistic_regression", lr_prob), ("linear_svm", svm_prob)]:
            if strategy_name == "threshold_adjust":
                threshold, sweep = select_threshold_by_precision_floor(
                    valid["label"], prob, precision_floor=float(cfg["threshold"]["precision_floor"]),
                    start=float(cfg["threshold"]["grid_start"]),
                    end=float(cfg["threshold"]["grid_end"]),
                    step=float(cfg["threshold"]["grid_step"]),
                )
                metrics = classification_report_at_threshold(valid["label"], prob, threshold=threshold)
                write_csv(sweep, results_dir / f"{model_name}_{strategy_name}_threshold_sweep.csv")
            elif strategy_name == "cost_sensitive":
                threshold = 0.5
                metrics = classification_report_at_threshold(valid["label"], prob, threshold=threshold)
            else:
                threshold = default_threshold
                metrics = classification_report_at_threshold(valid["label"], prob, threshold=threshold)

            rows.append(
                {
                    "strategy": strategy_name,
                    "model": model_name,
                    "threshold": float(threshold),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "macro_f1": metrics["macro_f1"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                }
            )

    summary = pd.DataFrame(rows)
    write_csv(summary, results_dir / "step6_imbalance_strategy_comparison.csv")
    write_json(
        {
            "precision_floor": float(cfg["threshold"]["precision_floor"]),
            "note": "Threshold adjusted strategy selects the best validation threshold under the precision floor.",
        },
        logs_dir / "step6_imbalance_log.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    compare_strategies(args.config)


if __name__ == "__main__":
    main()
