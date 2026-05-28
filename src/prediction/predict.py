from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.mindspore_dataset import TEXT_COLUMNS, pick_column
from src.evaluation.mindspore_metrics import sigmoid_np
from src.models.mindspore_model import build_cells, require_mindspore

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spam probability with a trained MindSpore model.")
    parser.add_argument("--model_dir", type=Path, default=ROOT / "outputs" / "mindspore" / "best")
    parser.add_argument("--text", default=None)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--output_csv", type=Path, default=ROOT / "outputs" / "mindspore" / "predictions.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device_target", choices=("CPU", "GPU", "Ascend"), default="CPU")
    args = parser.parse_args()
    if args.text is None and args.input_csv is None:
        parser.error("Pass --text or --input_csv")
    return args


def load_texts(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame]:
    if args.text is not None:
        return [args.text], pd.DataFrame({"clean_text": [args.text]})
    df = pd.read_csv(args.input_csv)
    text_col = pick_column(df.columns, TEXT_COLUMNS, "text")
    return df[text_col].fillna("").astype(str).tolist(), df.copy()


def main() -> None:
    args = parse_args()
    ms, Tensor, _, _ = require_mindspore()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    bundle = joblib.load(args.model_dir / "feature_bundle.joblib")
    texts, source = load_texts(args)
    features = bundle.transform(texts)

    Network, _ = build_cells(int(metadata["input_dim"]), int(metadata["hidden_dim"]), float(metadata["dropout"]))
    network = Network()
    params = ms.load_checkpoint(str(args.model_dir / "model.ckpt"))
    ms.load_param_into_net(network, params)
    network.set_train(False)

    logits = []
    for start in range(0, len(features), args.batch_size):
        batch = Tensor(features[start : start + args.batch_size].astype(np.float32), ms.float32)
        logits.append(network(batch).asnumpy())
    probabilities = sigmoid_np(np.concatenate(logits) / float(metadata.get("temperature", 1.0)))
    threshold = float(metadata["threshold"])
    pred = (probabilities >= threshold).astype(int)
    out = source.copy()
    out["probability_spam"] = probabilities
    out["pred_label"] = pred
    out["pred_name"] = np.where(pred == 1, "spam", "ham")
    out["confidence"] = np.where(pred == 1, probabilities, 1.0 - probabilities)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out[["probability_spam", "pred_name", "confidence"]].to_string(index=False))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
