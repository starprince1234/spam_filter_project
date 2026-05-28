from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.mindspore_dataset import create_generator_dataset, load_splits
from src.evaluation.mindspore_metrics import classification_report_at_threshold, fit_temperature, select_threshold_by_precision_floor, sigmoid_np
from src.features.mindspore_features import fit_feature_bundle
from src.models.mindspore_model import build_cells, predict_logits, require_mindspore

ROOT = Path(__file__).resolve().parents[2]


def _remove_readonly_and_retry(func, path, exc_info) -> None:
    if not isinstance(exc_info[1], PermissionError):
        raise exc_info[1]
    os.chmod(path, 0o666)
    func(path)


def prediction_frame(df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = (probabilities >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "label": df["label"].to_numpy(),
            "probability_spam": probabilities,
            "pred_label": pred,
            "confidence": np.where(pred == 1, probabilities, 1.0 - probabilities),
            "text_preview": df["clean_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 240),
        }
    )
    if "email_id" in df.columns:
        out.insert(0, "email_id", df["email_id"].to_numpy())
    return out


def train_network(args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray):
    ms, _, nn, _ = require_mindspore()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    ms.set_seed(args.seed)
    np.random.seed(args.seed)

    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = args.pos_weight if args.pos_weight is not None else negatives / positives

    Network, Loss = build_cells(x_train.shape[1], args.hidden_dim, args.dropout)
    network = Network()
    loss_fn = Loss(float(pos_weight))
    optimizer = nn.Adam(network.trainable_params(), learning_rate=args.learning_rate)

    def forward_fn(features, labels):
        logits = network(features)
        return loss_fn(logits, labels)

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    history = []

    for epoch in range(1, args.epochs + 1):
        network.set_train(True)
        losses = []
        train_ds = create_generator_dataset(x_train, y_train, args.batch_size, shuffle=True)
        for batch in train_ds.create_dict_iterator():
            loss, grads = grad_fn(batch["features"], batch["labels"])
            optimizer(grads)
            losses.append(float(loss.asnumpy()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"epoch={epoch} train_loss={mean_loss:.6f}")

    return network, history, float(pos_weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MindSpore spam filter.")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--valid_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--feature", choices=("word", "char", "structured", "hybrid", "word_svd", "char_svd", "hybrid_svd"), default="word")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "mindspore")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=0, help="0 gives logistic regression; >0 gives a one-hidden-layer MLP.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--precision_floor", type=float, default=0.95)
    parser.add_argument("--word_features", type=int, default=30000)
    parser.add_argument("--char_features", type=int, default=30000)
    parser.add_argument("--svd_components", type=int, default=200)
    parser.add_argument("--device_target", choices=("CPU", "GPU", "Ascend"), default="CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_calibrate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.feature
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = load_splits(ROOT, args.train_csv, args.valid_csv, args.test_csv)
    print(f"Loaded splits: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")
    x_train, feature_bundle = fit_feature_bundle(
        train_df["clean_text"],
        feature_type=args.feature,
        word_features=args.word_features,
        char_features=args.char_features,
        svd_components=args.svd_components,
    )
    x_valid = feature_bundle.transform(valid_df["clean_text"])
    x_test = feature_bundle.transform(test_df["clean_text"])
    y_train = train_df["label"].to_numpy().astype(np.float32)
    y_valid = valid_df["label"].to_numpy().astype(np.int32)
    y_test = test_df["label"].to_numpy().astype(np.int32)

    network, history, pos_weight = train_network(args, x_train, y_train)
    valid_logits = predict_logits(network, x_valid, batch_size=args.batch_size)
    temperature = 1.0 if args.no_calibrate else fit_temperature(valid_logits, y_valid)
    valid_prob = sigmoid_np(valid_logits / temperature)
    threshold, threshold_table = select_threshold_by_precision_floor(y_valid, valid_prob, precision_floor=args.precision_floor)
    valid_metrics = classification_report_at_threshold(y_valid, valid_prob, threshold=threshold)

    test_logits = predict_logits(network, x_test, batch_size=args.batch_size)
    test_prob = sigmoid_np(test_logits / temperature)
    test_metrics = classification_report_at_threshold(y_test, test_prob, threshold=threshold)

    ms, _, _, _ = require_mindspore()
    ms.save_checkpoint(network, str(run_dir / "model.ckpt"))
    joblib.dump(feature_bundle, run_dir / "feature_bundle.joblib")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    prediction_frame(valid_df, valid_prob, threshold).to_csv(run_dir / "valid_predictions.csv", index=False)
    prediction_frame(test_df, test_prob, threshold).to_csv(run_dir / "test_predictions.csv", index=False)
    threshold_table.to_csv(run_dir / "threshold_sweep.csv", index=False)

    metadata = {
        "feature_type": args.feature,
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "pos_weight": float(pos_weight),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    best_dir = args.output_dir / "best"
    staging_best_dir = args.output_dir / f"best.tmp.{uuid.uuid4().hex}"
    if staging_best_dir.exists():
        shutil.rmtree(staging_best_dir, onerror=_remove_readonly_and_retry)
    shutil.copytree(run_dir, staging_best_dir)
    if best_dir.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(best_dir, onerror=_remove_readonly_and_retry)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(1.0)
    staging_best_dir.replace(best_dir)
    print(json.dumps({"threshold": threshold, "valid": valid_metrics, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
