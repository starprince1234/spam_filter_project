from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import TEXT_COLUMNS, pick_column
from src.mindspore_model import build_cells, require_mindspore, sigmoid_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spam probability with a trained MindSpore model.")
    parser.add_argument("--model_dir", type=Path, default=ROOT / "outputs" / "best")
    parser.add_argument("--input_csv", default=None, help="CSV containing a clean_text/text/body/content column.")
    parser.add_argument("--text", default=None, help="Single email text.")
    parser.add_argument("--output_csv", type=Path, default=ROOT / "outputs" / "predictions.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device_target", default=None, choices=(None, "CPU", "GPU", "Ascend"))
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame]:
    if args.text is None and args.input_csv is None:
        raise ValueError("Pass --text or --input_csv.")
    if args.text is not None:
        return [args.text], pd.DataFrame({"clean_text": [args.text]})
    df = pd.read_csv(args.input_csv)
    text_col = pick_column(df.columns, TEXT_COLUMNS, "text")
    return df[text_col].fillna("").astype(str).tolist(), df.copy()


def main() -> None:
    args = parse_args()
    ms, Tensor, _, _ = require_mindspore()
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    device_target = args.device_target or metadata.get("device_target", "CPU")
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=device_target)

    texts, source = load_texts(args)
    bundle = joblib.load(args.model_dir / "feature_bundle.joblib")
    x = bundle.transform(texts).astype("float32")

    SpamNet, _ = build_cells(int(metadata["input_dim"]), int(metadata["hidden_dim"]), float(metadata["dropout"]))
    model = SpamNet()
    params = ms.load_checkpoint(str(args.model_dir / "model.ckpt"))
    ms.load_param_into_net(model, params)
    model.set_train(False)

    logits = []
    for start in range(0, len(x), args.batch_size):
        batch = Tensor(x[start : start + args.batch_size], ms.float32)
        logits.append(model(batch).asnumpy())
    prob = sigmoid_np(np.concatenate(logits) / float(metadata.get("temperature", 1.0)))
    pred = (prob >= float(metadata["threshold"])).astype(int)

    out = source.copy()
    out["probability_spam"] = prob
    out["pred_label"] = pred
    out["pred_name"] = np.where(pred == 1, "spam", "ham")
    out["confidence"] = np.where(pred == 1, prob, 1.0 - prob)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out[["probability_spam", "pred_name", "confidence"]].to_string(index=False))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()

