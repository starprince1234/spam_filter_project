from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.thresholds import select_threshold_by_precision_floor, threshold_sweep
from src.utils.io_utils import ensure_dir, load_yaml, write_csv, write_json


def run_threshold_analysis(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    results_dir = root / cfg["paths"]["results"]
    logs_dir = root / cfg["paths"]["logs"]
    ensure_dir(results_dir)
    ensure_dir(logs_dir)

    rows = []
    for model_name in ["logistic_regression", "linear_svm"]:
        pred_path = results_dir / f"{model_name}_calibrated_valid_predictions.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            y_true = pred["y_true"].to_numpy()
            y_prob = pred["prob_after"].to_numpy()
        else:
            pred = pd.read_csv(results_dir / f"{model_name}_valid_predictions.csv")
            y_true = pred["label"].to_numpy()
            y_prob = pred["probability"].to_numpy()

        threshold, table = select_threshold_by_precision_floor(
            y_true,
            y_prob,
            precision_floor=float(cfg["threshold"]["precision_floor"]),
            start=float(cfg["threshold"]["grid_start"]),
            end=float(cfg["threshold"]["grid_end"]),
            step=float(cfg["threshold"]["grid_step"]),
        )
        sweep = threshold_sweep(
            y_true,
            y_prob,
            start=float(cfg["threshold"]["grid_start"]),
            end=float(cfg["threshold"]["grid_end"]),
            step=float(cfg["threshold"]["grid_step"]),
        )
        write_csv(sweep, results_dir / f"{model_name}_threshold_sweep.csv")
        rows.append(
            {
                "model": model_name,
                "selected_threshold": threshold,
                "best_precision": float(table[table["threshold"] == threshold]["precision"].iloc[0]),
                "best_recall": float(table[table["threshold"] == threshold]["recall"].iloc[0]),
                "best_f1": float(table[table["threshold"] == threshold]["f1"].iloc[0]),
            }
        )

    write_csv(pd.DataFrame(rows), results_dir / "step8_threshold_analysis.csv")
    write_json({"precision_floor": float(cfg["threshold"]["precision_floor"])}, logs_dir / "step8_threshold_analysis_log.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_threshold_analysis(args.config)


if __name__ == "__main__":
    main()
