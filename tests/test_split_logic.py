from __future__ import annotations

import pandas as pd

from src.data.build_dataset import split_by_group_and_label


def test_split_keeps_all_splits_with_both_labels():
    df = pd.DataFrame(
        {
            "group_key": [f"g{i}" for i in range(100)],
            "thread_group": [f"g{i}" for i in range(100)],
            "label": [0] * 80 + [1] * 20,
        }
    )
    out = split_by_group_and_label(df, seed=42)
    for split in ["train", "valid", "test"]:
        labels = set(out.loc[out["split"] == split, "label"].tolist())
        assert labels == {0, 1}
