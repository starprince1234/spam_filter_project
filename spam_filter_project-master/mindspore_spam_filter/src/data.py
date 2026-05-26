from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEXT_COLUMNS = ("clean_text", "text", "body", "content", "message", "email_text")
LABEL_COLUMNS = ("label", "is_spam", "target", "y")
SPLIT_COLUMNS = ("split", "dataset_split", "fold")


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except IndexError:
        return False
    return first.startswith("version https://git-lfs.github.com/spec/v1")


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the real dataset. "
            "Run `git lfs pull` or replace it with the actual CSV before training."
        )
    return pd.read_csv(path)


def pick_column(columns: Iterable[str], candidates: Iterable[str], role: str) -> str:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ValueError(f"Could not find a {role} column. Tried: {', '.join(candidates)}")


def normalize_labels(values: pd.Series) -> np.ndarray:
    def one(v) -> int:
        if pd.isna(v):
            raise ValueError("Label column contains missing values.")
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"spam", "1", "true", "yes", "junk"}:
                return 1
            if s in {"ham", "0", "false", "no", "normal", "nonspam", "non-spam"}:
                return 0
        return int(v)

    y = values.map(one).astype("int32").to_numpy()
    bad = sorted(set(y.tolist()) - {0, 1})
    if bad:
        raise ValueError(f"Labels must be binary 0/1 after normalization, got {bad}.")
    return y


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
    main_csv: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = project_root / "data" / "splits"
    if train_csv or valid_csv or test_csv:
        if not (train_csv and valid_csv and test_csv):
            raise ValueError("Pass all of --train_csv, --valid_csv and --test_csv, or pass none of them.")
        train = read_csv_checked(Path(train_csv))
        valid = read_csv_checked(Path(valid_csv))
        test = read_csv_checked(Path(test_csv))
        return prepare_frame(train), prepare_frame(valid), prepare_frame(test)

    default_train = split_dir / "train.csv"
    default_valid = split_dir / "valid.csv"
    default_test = split_dir / "test.csv"
    if all(p.exists() and not is_lfs_pointer(p) for p in (default_train, default_valid, default_test)):
        return (
            prepare_frame(read_csv_checked(default_train)),
            prepare_frame(read_csv_checked(default_valid)),
            prepare_frame(read_csv_checked(default_test)),
        )

    main_path = Path(main_csv) if main_csv else project_root / "data" / "cleaned" / "main_dataset_5000.csv"
    df = read_csv_checked(main_path)
    split_col = pick_column(df.columns, SPLIT_COLUMNS, "split")
    train = df[df[split_col].astype(str).str.lower().eq("train")]
    valid = df[df[split_col].astype(str).str.lower().isin(["valid", "validation", "val"])]
    test = df[df[split_col].astype(str).str.lower().eq("test")]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"Split column {split_col!r} did not contain train/valid/test rows.")
    return prepare_frame(train), prepare_frame(valid), prepare_frame(test)

