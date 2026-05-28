from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TEXT_COLUMNS = ("clean_text", "text", "body", "content", "message", "email_text")
LABEL_COLUMNS = ("label", "is_spam", "target", "y")


class ArrayDataset:
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def pick_column(columns: list[str] | pd.Index, candidates: tuple[str, ...], role: str) -> str:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(f"Could not find a {role} column. Tried: {', '.join(candidates)}")


def normalize_labels(values: pd.Series) -> np.ndarray:
    def normalize_one(value: object) -> int:
        if pd.isna(value):
            raise ValueError("Label column contains missing values")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"spam", "1", "true", "yes", "junk"}:
                return 1
            if normalized in {"ham", "0", "false", "no", "normal", "nonspam", "non-spam"}:
                return 0
        return int(value)

    labels = values.map(normalize_one).astype("int32").to_numpy()
    unexpected = sorted(set(labels.tolist()) - {0, 1})
    if unexpected:
        raise ValueError(f"Labels must be binary 0/1 after normalization, got {unexpected}")
    return labels


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    text_col = pick_column(df.columns, TEXT_COLUMNS, "text")
    label_col = pick_column(df.columns, LABEL_COLUMNS, "label")
    out = df.copy()
    out["clean_text"] = out[text_col].fillna("").astype(str)
    out["label"] = normalize_labels(out[label_col])
    return out


def load_splits(
    project_root: Path,
    train_csv: str | None = None,
    valid_csv: str | None = None,
    test_csv: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_csv or valid_csv or test_csv:
        if not (train_csv and valid_csv and test_csv):
            raise ValueError("Pass all of --train_csv, --valid_csv and --test_csv, or pass none of them")
        return (
            prepare_frame(pd.read_csv(train_csv)),
            prepare_frame(pd.read_csv(valid_csv)),
            prepare_frame(pd.read_csv(test_csv)),
        )

    split_dir = project_root / "data" / "splits"
    return (
        prepare_frame(pd.read_csv(split_dir / "train.csv")),
        prepare_frame(pd.read_csv(split_dir / "valid.csv")),
        prepare_frame(pd.read_csv(split_dir / "test.csv")),
    )


def create_generator_dataset(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    import mindspore.dataset as ds

    source = ArrayDataset(features, labels)
    dataset = ds.GeneratorDataset(source, column_names=["features", "labels"], shuffle=shuffle)
    return dataset.batch(batch_size, drop_remainder=False)


def iter_batches(features: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(features), batch_size):
        yield features[start : start + batch_size]
