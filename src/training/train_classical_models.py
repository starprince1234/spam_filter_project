from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.evaluation.metrics import classification_report_at_threshold
from src.features.text_features import extract_structured_features
from src.utils.io_utils import ensure_dir, load_yaml, write_csv, write_json


def _load_splits(root: Path):
    train = pd.read_csv(root / "data/splits/train.csv")
    valid = pd.read_csv(root / "data/splits/valid.csv")
    test = pd.read_csv(root / "data/splits/test.csv")
    return train, valid, test


def _fit_word_vectorizer(train_texts):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=30000)
    X_train = vec.fit_transform(train_texts)
    return vec, X_train


def train_lr(train, valid):
    vec, X_train = _fit_word_vectorizer(train["clean_text"].fillna(""))
    X_valid = vec.transform(valid["clean_text"].fillna(""))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
    clf.fit(X_train, train["label"])
    valid_prob = clf.predict_proba(X_valid)[:, 1]
    return {"model": clf, "vectorizer": vec, "valid_prob": valid_prob, "name": "logistic_regression"}


def train_nb(train, valid):
    vec, X_train = _fit_word_vectorizer(train["clean_text"].fillna(""))
    X_valid = vec.transform(valid["clean_text"].fillna(""))
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train, train["label"])
    valid_prob = clf.predict_proba(X_valid)[:, 1]
    return {"model": clf, "vectorizer": vec, "valid_prob": valid_prob, "name": "multinomial_nb"}


def train_svm(train, valid):
    vec, X_train = _fit_word_vectorizer(train["clean_text"].fillna(""))
    X_valid = vec.transform(valid["clean_text"].fillna(""))
    base = LinearSVC(class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X_train, train["label"])
    valid_prob = clf.predict_proba(X_valid)[:, 1]
    return {"model": clf, "vectorizer": vec, "valid_prob": valid_prob, "name": "linear_svm"}


def train_all(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    train, valid, test = _load_splits(root)
    models_dir = root / "models" / "raw"
    results_dir = root / "experiments" / "results"
    ensure_dir(models_dir)
    ensure_dir(results_dir)

    train_funcs = [train_nb, train_lr, train_svm]
    summary = []
    for fn in train_funcs:
        out = fn(train, valid)
        name = out["name"]
        model = out["model"]
        vec = out["vectorizer"]
        valid_prob = out["valid_prob"]
        valid_metrics = classification_report_at_threshold(valid["label"], valid_prob, threshold=0.5)
        summary.append({"model": name, **{k: v for k, v in valid_metrics.items() if k != "confusion_matrix"}})
        joblib.dump({"model": model, "vectorizer": vec}, models_dir / f"{name}.joblib")
        pred_df = valid[["email_id", "label", "source_dataset"]].copy()
        pred_df["probability"] = valid_prob
        pred_df["pred_label"] = (pred_df["probability"] >= 0.5).astype(int)
        write_csv(pred_df, results_dir / f"{name}_valid_predictions.csv")
        write_json(valid_metrics, results_dir / f"{name}_valid_metrics.json")

    write_csv(pd.DataFrame(summary), results_dir / "step5_model_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_all(args.config)


if __name__ == "__main__":
    main()
