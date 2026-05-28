from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.data.mindspore_dataset import load_splits
from src.evaluation.mindspore_metrics import classification_report_at_threshold, select_threshold_by_precision_floor, sigmoid_np
from src.models.mindspore_model import build_cells, predict_logits, require_mindspore

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MindSpore spam filter.")
    parser.add_argument("--model_dir", type=Path, default=ROOT / "outputs" / "mindspore" / "best")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device_target", choices=("CPU", "GPU", "Ascend"), default="CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ms, _, _, _ = require_mindspore()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device_target)
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))
    _, valid_df, test_df = load_splits(ROOT)
    bundle = joblib.load(args.model_dir / "feature_bundle.joblib")
    x_valid = bundle.transform(valid_df["clean_text"])
    x_test = bundle.transform(test_df["clean_text"])

    Network, _ = build_cells(int(metadata["input_dim"]), int(metadata["hidden_dim"]), float(metadata["dropout"]))
    network = Network()
    params = ms.load_checkpoint(str(args.model_dir / "model.ckpt"))
    ms.load_param_into_net(network, params)

    valid_prob = sigmoid_np(predict_logits(network, x_valid, args.batch_size) / float(metadata.get("temperature", 1.0)))
    threshold, _ = select_threshold_by_precision_floor(valid_df["label"].to_numpy(), valid_prob)
    test_prob = sigmoid_np(predict_logits(network, x_test, args.batch_size) / float(metadata.get("temperature", 1.0)))
    report = classification_report_at_threshold(test_df["label"].to_numpy(), test_prob, threshold)
    print(json.dumps({"threshold": threshold, "test_metrics": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
