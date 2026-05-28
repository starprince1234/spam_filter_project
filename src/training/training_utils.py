from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_report_at_threshold
from src.utils.io_utils import ensure_dir, write_csv, write_json


@dataclass
class TrainResult:
    model_name: str
    threshold: float
    metrics: dict[str, Any]


def save_model(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)


def save_predictions(df: pd.DataFrame, path: str | Path) -> None:
    write_csv(df, path)


def evaluate_model(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    return classification_report_at_threshold(y_true, y_prob, threshold=threshold)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    write_json(metrics, path)
