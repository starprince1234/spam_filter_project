from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.utils.io_utils import ensure_dir, write_csv, write_json


def classify_error_case(row: pd.Series) -> str:
    text = f"{row.get('subject', '')} {row.get('clean_text', '')}".lower()
    if row.get("label") == 0 and row.get("pred_label") == 1:
        if any(k in text for k in ["free", "winner", "offer", "urgent", "discount"]):
            return "营销类邮件被误判"
        if any(k in text for k in ["system", "notification", "alert", "notice", "reminder"]):
            return "系统通知邮件被误判"
        if row.get("has_html", 0):
            return "营销类邮件被误判"
        if len(text.split()) < 8:
            return "短文本邮件误判"
        return "系统通知邮件被误判"
    if row.get("label") == 1 and row.get("pred_label") == 0:
        if row.get("has_html", 0):
            return "HTML 伪装邮件漏判"
        if any(k in text for k in ["free", "winner", "offer", "urgent", "discount", "click", "verify"]):
            return "钓鱼邮件漏判"
        if any(ch.isdigit() for ch in text) and len(text.split()) < 10:
            return "伪装正常邮件漏判"
        if any(repeat in text for repeat in ["looo", "winnnn", "frree"]):
            return "拼写变形邮件漏判"
        return "短文本邮件误判"
    return "正确"


def summarize_misclassifications(pred_df: pd.DataFrame) -> pd.DataFrame:
    errors = pred_df[pred_df["label"] != pred_df["pred_label"]].copy()
    if errors.empty:
        return pd.DataFrame(columns=list(pred_df.columns) + ["error_type"])
    errors["error_type"] = errors.apply(classify_error_case, axis=1)
    return errors


def _export_lr_top_features(root: Path, out_dir: Path, top_k: int = 30) -> None:
    lr_obj = joblib.load(root / "models" / "raw" / "logistic_regression.joblib")
    lr = lr_obj["model"]
    vec = lr_obj["vectorizer"]
    coef = lr.coef_.ravel()
    names = vec.get_feature_names_out()

    pos_idx = coef.argsort()[-top_k:][::-1]
    neg_idx = coef.argsort()[:top_k]
    top_pos = pd.DataFrame({"feature": names[pos_idx], "weight": coef[pos_idx], "direction": "spam_positive"})
    top_neg = pd.DataFrame({"feature": names[neg_idx], "weight": coef[neg_idx], "direction": "ham_negative"})
    write_csv(pd.concat([top_pos, top_neg], axis=0).reset_index(drop=True), out_dir / "step9_lr_top_features.csv")


def export_error_analysis(root: Path, model_name: str = "logistic_regression", threshold: float = 0.84) -> None:
    results_dir = root / "experiments" / "results"
    error_dir = root / "outputs" / "error_analysis"
    tables_dir = root / "outputs" / "tables"
    splits_dir = root / "data" / "splits"
    ensure_dir(error_dir)
    ensure_dir(tables_dir)

    pred = pd.read_csv(results_dir / f"{model_name}_calibrated_valid_predictions.csv")
    valid = pd.read_csv(splits_dir / "valid.csv")
    merged = pred.merge(
        valid[["email_id", "subject", "clean_text", "has_html", "source_dataset", "sender"]],
        how="left",
        on="email_id",
    )
    merged = merged.rename(columns={"y_true": "label", "prob_after": "probability"})
    merged["pred_label"] = (merged["probability"] >= threshold).astype(int)
    errors = summarize_misclassifications(merged)
    if not errors.empty:
        errors = errors.sort_values("probability", ascending=False).reset_index(drop=True)
        errors.to_csv(error_dir / f"{model_name}_validation_errors.csv", index=False, encoding="utf-8")
        write_csv(
            errors[
                [
                    "email_id",
                    "label",
                    "pred_label",
                    "probability",
                    "error_type",
                    "subject",
                    "clean_text",
                    "source_dataset",
                    "sender",
                ]
            ].head(50),
            error_dir / "step9_typical_error_cases.csv",
        )
        write_csv(errors[["label", "pred_label", "error_type"]].groupby("error_type").size().reset_index(name="count"), tables_dir / "step9_error_type_summary.csv")
    else:
        write_csv(pd.DataFrame(columns=["error_type", "count"]), tables_dir / "step9_error_type_summary.csv")
        write_csv(pd.DataFrame(columns=["email_id", "label", "pred_label", "probability", "error_type", "subject", "clean_text"]), error_dir / "step9_typical_error_cases.csv")

    _export_lr_top_features(root, tables_dir)
    write_json({"model": model_name, "errors": int(len(errors)), "threshold": threshold}, error_dir / "step9_error_analysis_log.json")
