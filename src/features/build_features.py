from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

from src.features.text_features import (
    build_char_only,
    build_hybrid_features,
    build_structured_only,
    build_text_svd_features,
    build_word_only,
    extract_structured_features,
)
from src.utils.io_utils import ensure_dir, load_yaml, write_csv, write_json


def _save_sparse_matrix(X, path: Path):
    ensure_dir(path.parent)
    sparse.save_npz(path, X)


def build_features(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    splits_dir = root / cfg["paths"]["splits"]
    results_dir = root / cfg["paths"]["results"]
    tables_dir = root / cfg["paths"]["tables"]
    ensure_dir(results_dir)
    ensure_dir(tables_dir)

    train = pd.read_csv(splits_dir / "train.csv")
    valid = pd.read_csv(splits_dir / "valid.csv")
    test = pd.read_csv(splits_dir / "test.csv")
    all_df = pd.concat([train, valid, test], axis=0).reset_index(drop=True)

    structured = extract_structured_features(all_df)
    structured_train = structured.iloc[: len(train)]
    structured_valid = structured.iloc[len(train) : len(train) + len(valid)]
    structured_test = structured.iloc[len(train) + len(valid) :]

    word_X_train, word_vec = build_word_only(train["clean_text"].fillna(""))
    word_X_valid = word_vec.transform(valid["clean_text"].fillna(""))
    word_X_test = word_vec.transform(test["clean_text"].fillna(""))

    char_X_train, char_vec = build_char_only(train["clean_text"].fillna(""))
    char_X_valid = char_vec.transform(valid["clean_text"].fillna(""))
    char_X_test = char_vec.transform(test["clean_text"].fillna(""))

    struct_X_train, struct_scaler = build_structured_only(structured_train)
    struct_X_valid = struct_scaler.transform(structured_valid.fillna(0.0).values)
    struct_X_test = struct_scaler.transform(structured_test.fillna(0.0).values)

    hybrid_X_train, hybrid_artifacts = build_hybrid_features(train["clean_text"].fillna(""), structured_train)
    hybrid_X_valid = sparse.hstack(
        [
            hybrid_artifacts["word"].transform(valid["clean_text"].fillna("")),
            hybrid_artifacts["char"].transform(valid["clean_text"].fillna("")),
            sparse.csr_matrix(hybrid_artifacts["scaler"].transform(structured_valid.fillna(0.0).values)),
        ]
    )
    hybrid_X_test = sparse.hstack(
        [
            hybrid_artifacts["word"].transform(test["clean_text"].fillna("")),
            hybrid_artifacts["char"].transform(test["clean_text"].fillna("")),
            sparse.csr_matrix(hybrid_artifacts["scaler"].transform(structured_test.fillna(0.0).values)),
        ]
    )

    svd_X_train, svd_artifacts = build_text_svd_features(train["clean_text"].fillna(""), structured_train)
    svd_X_valid = svd_artifacts["svd"].transform(svd_artifacts["word"].transform(valid["clean_text"].fillna("")))
    svd_X_valid = __import__("numpy").hstack([svd_X_valid, struct_scaler.transform(structured_valid.fillna(0.0).values)])
    svd_X_test = svd_artifacts["svd"].transform(svd_artifacts["word"].transform(test["clean_text"].fillna("")))
    svd_X_test = __import__("numpy").hstack([svd_X_test, struct_scaler.transform(structured_test.fillna(0.0).values)])

    feature_dir = root / "outputs" / "features"
    ensure_dir(feature_dir)

    _save_sparse_matrix(word_X_train, feature_dir / "word_train.npz")
    _save_sparse_matrix(word_X_valid, feature_dir / "word_valid.npz")
    _save_sparse_matrix(word_X_test, feature_dir / "word_test.npz")
    _save_sparse_matrix(char_X_train, feature_dir / "char_train.npz")
    _save_sparse_matrix(char_X_valid, feature_dir / "char_valid.npz")
    _save_sparse_matrix(char_X_test, feature_dir / "char_test.npz")
    write_csv(structured_train, feature_dir / "structured_train.csv")
    write_csv(structured_valid, feature_dir / "structured_valid.csv")
    write_csv(structured_test, feature_dir / "structured_test.csv")
    _save_sparse_matrix(hybrid_X_train, feature_dir / "hybrid_train.npz")
    _save_sparse_matrix(hybrid_X_valid, feature_dir / "hybrid_valid.npz")
    _save_sparse_matrix(hybrid_X_test, feature_dir / "hybrid_test.npz")

    joblib.dump(word_vec, root / "models" / "raw" / "word_tfidf.joblib")
    joblib.dump(char_vec, root / "models" / "raw" / "char_tfidf.joblib")
    joblib.dump(hybrid_artifacts, root / "models" / "raw" / "hybrid_artifacts.joblib")
    joblib.dump(struct_scaler, root / "models" / "raw" / "structured_scaler.joblib")

    stats = {
        "word_dim": int(word_X_train.shape[1]),
        "char_dim": int(char_X_train.shape[1]),
        "structured_dim": int(structured_train.shape[1]),
        "hybrid_dim": int(hybrid_X_train.shape[1]),
        "svd_dim": int(svd_X_train.shape[1]),
        "train_size": int(len(train)),
        "valid_size": int(len(valid)),
        "test_size": int(len(test)),
    }
    write_json(stats, tables_dir / "step4_feature_stats.json")
    write_csv(pd.DataFrame(list(stats.items()), columns=["metric", "value"]), tables_dir / "step4_feature_summary.csv")

    sample_check = pd.concat(
        [
            train[["email_id", "label", "source_dataset"]].head(5).reset_index(drop=True),
            pd.DataFrame({"feature_block": ["word", "char", "structured", "hybrid", "svd"]}),
        ],
        axis=1,
    )
    write_csv(sample_check, tables_dir / "step4_feature_sample_check.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_features(args.config)


if __name__ == "__main__":
    main()
