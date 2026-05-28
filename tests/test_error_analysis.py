from __future__ import annotations

import numpy as np
import pandas as pd

from src.explainability.error_analysis import classify_error_case, summarize_misclassifications


def test_classify_error_case_labels_false_negative():
    row = pd.Series({"label": 1, "pred_label": 0, "subject": "win free money", "clean_text": "click now", "has_html": 1})
    assert classify_error_case(row) in {"钓鱼邮件漏判", "HTML 伪装邮件漏判", "伪装正常邮件漏判"}


def test_summarize_misclassifications_returns_dataframe():
    df = pd.DataFrame(
        {
            "email_id": ["a", "b"],
            "label": [0, 1],
            "pred_label": [1, 0],
            "subject": ["meeting", "free offer"],
            "clean_text": ["project update", "click now"],
            "has_html": [0, 1],
        }
    )
    out = summarize_misclassifications(df)
    assert isinstance(out, pd.DataFrame)
    assert not out.empty
