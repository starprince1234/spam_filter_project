from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.calibration.calibration_utils import apply_temperature_scaling, calibrate_probabilities_sigmoid
from src.evaluation.metrics import classification_report_at_threshold
from src.utils.io_utils import ensure_dir, load_yaml, write_csv, write_json


def _load_results(root: Path, model_name: str):
    valid = pd.read_csv(root / "experiments/results" / f"{model_name}_valid_predictions.csv")
    model = joblib.load(root / "models/raw" / f"{model_name}.joblib")
    return valid, model


def run_calibration(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    results_dir = root / cfg["paths"]["results"]
    logs_dir = root / cfg["paths"]["logs"]
    models_cal = root / cfg["paths"]["models_calibrated"]
    figures_dir = root / cfg["paths"]["figures"]
    ensure_dir(results_dir)
    ensure_dir(logs_dir)
    ensure_dir(models_cal)
    ensure_dir(figures_dir)

    rows = []
    for model_name in ["logistic_regression", "linear_svm"]:
        valid_pred, model_obj = _load_results(root, model_name)
        y_true = valid_pred["label"].to_numpy()
        y_prob = valid_pred["probability"].to_numpy()
        before = classification_report_at_threshold(y_true, y_prob, threshold=0.5)

        if model_name == "linear_svm":
            calibrated_prob, calibrator = calibrate_probabilities_sigmoid(y_true, y_prob)
            calibrated_type = "sigmoid"
        else:
            calibrated_prob, temp = apply_temperature_scaling(y_true, y_prob)
            calibrator = {"temperature": temp}
            calibrated_type = "temperature"

        after = classification_report_at_threshold(y_true, calibrated_prob, threshold=0.5)
        write_csv(
            pd.DataFrame(
                {
                    "email_id": valid_pred["email_id"],
                    "y_true": y_true,
                    "prob_before": y_prob,
                    "prob_after": calibrated_prob,
                }
            ),
            results_dir / f"{model_name}_calibrated_valid_predictions.csv",
        )
        joblib.dump({"model": model_obj, "calibrator": calibrator, "calibration_type": calibrated_type}, models_cal / f"{model_name}_calibrated.joblib")
        write_json({"before": before, "after": after, "calibration_type": calibrated_type}, logs_dir / f"{model_name}_calibration_log.json")
        rows.append(
            {
                "model": model_name,
                "calibration_type": calibrated_type,
                "brier_before": before["brier"],
                "brier_after": after["brier"],
                "ece_before": before["ece"],
                "ece_after": after["ece"],
                "pr_auc_before": before["pr_auc"],
                "pr_auc_after": after["pr_auc"],
            }
        )

    write_csv(pd.DataFrame(rows), results_dir / "step7_calibration_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_calibration(args.config)


if __name__ == "__main__":
    main()
