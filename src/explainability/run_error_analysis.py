from __future__ import annotations

import argparse
from pathlib import Path

from src.explainability.error_analysis import export_error_analysis
from src.utils.io_utils import load_yaml


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    root = Path(config_path).resolve().parents[1]
    threshold = 0.84
    export_error_analysis(root, model_name=cfg["models"]["deployment_default"].replace("calibrated_", ""), threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
