from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.data import load_splits
from src.features import fit_feature_bundle
from src.metrics import binary_metrics, select_threshold
from src.mindspore_model import predict_logits, require_mindspore, sigmoid_np, train_model
from src.reporting import write_json, write_practice_report


FEATURES = ("structured", "word_svd", "char_svd", "hybrid")


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    y = y_true.astype("float64")

    def nll(log_t: float) -> float:
        temp = float(np.exp(log_t))
        prob = np.clip(sigmoid_np(logits / temp), 1e-7, 1.0 - 1e-7)
        return float(-(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)).mean())

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    return float(np.exp(result.x))


def prediction_frame(df: pd.DataFrame, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = (prob >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "label": df["label"].to_numpy(),
            "probability_spam": prob,
            "pred_label": pred,
            "confidence": np.where(pred == 1, prob, 1.0 - prob),
            "text_preview": df["clean_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 240),
        }
    )
    if "email_id" in df.columns:
        out.insert(0, "email_id", df["email_id"].to_numpy())
    return out


def train_one_feature(args: argparse.Namespace, feature_type: str, train_df, valid_df, test_df) -> dict:
    run_dir = args.output_dir / feature_type
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train, bundle = fit_feature_bundle(
        train_df["clean_text"],
        feature_type=feature_type,
        word_features=args.word_features,
        char_features=args.char_features,
        svd_components=args.svd_components,
    )
    x_valid = bundle.transform(valid_df["clean_text"])
    x_test = bundle.transform(test_df["clean_text"])
    y_train = train_df["label"].to_numpy().astype("int32")
    y_valid = valid_df["label"].to_numpy().astype("int32")
    y_test = test_df["label"].to_numpy().astype("int32")

    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = args.pos_weight if args.pos_weight is not None else negatives / positives

    model, history, valid_logits = train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pos_weight=pos_weight,
        device_target=args.device_target,
    )

    temperature = fit_temperature(valid_logits, y_valid) if args.calibrate else 1.0
    valid_prob = sigmoid_np(valid_logits / temperature)
    threshold, valid_metrics = select_threshold(
        y_valid,
        valid_prob,
        min_precision=args.min_precision,
        metric=args.threshold_metric,
    )

    test_logits = predict_logits(model, x_test, batch_size=args.batch_size, device_target=args.device_target)
    test_prob = sigmoid_np(test_logits / temperature)
    test_metrics = binary_metrics(y_test, test_prob, threshold=threshold)

    valid_pred = prediction_frame(valid_df, valid_prob, threshold)
    test_pred = prediction_frame(test_df, test_prob, threshold)
    valid_pred.to_csv(run_dir / "valid_predictions.csv", index=False)
    test_pred.to_csv(run_dir / "test_predictions.csv", index=False)
    errors = test_pred[test_pred["label"] != test_pred["pred_label"]].copy()
    errors.sort_values("confidence", ascending=False).to_csv(run_dir / "test_errors.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    ms, _, _, _ = require_mindspore()
    ms.save_checkpoint(model, str(run_dir / "model.ckpt"))
    joblib.dump(bundle, run_dir / "feature_bundle.joblib")

    metadata = {
        "feature_type": feature_type,
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "pos_weight": float(pos_weight),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "device_target": args.device_target,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    write_json(run_dir / "metadata.json", metadata)

    row = {
        "feature": feature_type,
        "input_dim": int(x_train.shape[1]),
        "threshold": threshold,
        "temperature": temperature,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "macro_f1": test_metrics["macro_f1"],
        "pr_auc": test_metrics["pr_auc"],
        "roc_auc": test_metrics["roc_auc"],
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "model_dir": str(run_dir),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MindSpore spam filter.")
    parser.add_argument("--feature", choices=FEATURES + ("all",), default="all")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--valid_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--main_csv", default=None)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128, help="Use 0 for MindSpore logistic regression.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--min_precision", type=float, default=0.95)
    parser.add_argument("--threshold_metric", choices=("f1", "macro_f1", "recall", "precision"), default="f1")
    parser.add_argument("--word_features", type=int, default=30000)
    parser.add_argument("--char_features", type=int, default=30000)
    parser.add_argument("--svd_components", type=int, default=200)
    parser.add_argument("--device_target", default="CPU", choices=("CPU", "GPU", "Ascend"))
    parser.add_argument("--calibrate", action="store_true", help="Fit temperature scaling on validation logits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_mindspore()

    train_df, valid_df, test_df = load_splits(
        PROJECT_ROOT,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        main_csv=args.main_csv,
    )
    features = FEATURES if args.feature == "all" else (args.feature,)
    rows = []
    for feature_type in features:
        print(f"Training feature set: {feature_type}")
        rows.append(train_one_feature(args, feature_type, train_df, valid_df, test_df))

    comparison = pd.DataFrame(rows).sort_values(["f1", "macro_f1"], ascending=False)
    comparison_path = args.output_dir / "feature_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    best = comparison.iloc[0].to_dict()
    best_dir = Path(best["model_dir"])
    write_json(args.output_dir / "best_model_summary.json", best)
    if (args.output_dir / "best").exists():
        shutil.rmtree(args.output_dir / "best")
    shutil.copytree(best_dir, args.output_dir / "best")

    write_practice_report(
        output_path=args.output_dir / "mindspore_practice_report.md",
        comparison_csv=comparison_path,
        best_feature=str(best["feature"]),
        threshold=float(best["threshold"]),
        metrics=json.loads((best_dir / "metadata.json").read_text(encoding="utf-8"))["test_metrics"],
        error_csv=best_dir / "test_errors.csv",
    )
    print(f"Done. Best model: {best_dir}")
    print(f"Report: {args.output_dir / 'mindspore_practice_report.md'}")


if __name__ == "__main__":
    main()

