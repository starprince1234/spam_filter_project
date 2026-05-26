from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.email_parser import parse_eml_file
from src.explainability.error_analysis import classify_error_case
from src.utils.io_utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CFG = load_yaml(ROOT / "configs" / "config.yaml")
RESULTS_DIR = ROOT / "experiments" / "results"
ERROR_DIR = ROOT / "outputs" / "error_analysis"
TABLES_DIR = ROOT / "outputs" / "tables"
MODELS_DIR = ROOT / "models" / "calibrated"
PRED_DIR = ROOT / "outputs" / "predictions"


@st.cache_resource
def load_default_model():
    path = MODELS_DIR / "logistic_regression_calibrated.joblib"
    if path.exists():
        return joblib.load(path)
    return joblib.load(ROOT / "models" / "raw" / "logistic_regression.joblib")


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "安全"
    if prob < 0.70:
        return "可疑"
    return "垃圾"


def predict_text(text: str) -> dict:
    bundle = load_default_model()
    if isinstance(bundle, dict) and "model" in bundle and isinstance(bundle["model"], dict):
        base = bundle["model"]
        model = base["model"]
        vec = base["vectorizer"]
        calibration_type = bundle.get("calibration_type")
        calibrator = bundle.get("calibrator")
    else:
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        vec = bundle.get("vectorizer") if isinstance(bundle, dict) else None
        calibration_type = None
        calibrator = None
    if vec is None:
        return {"probability": 0.0, "label": "不可用"}
    prob = float(model.predict_proba(vec.transform([text]))[:, 1][0])
    if calibration_type == "temperature" and isinstance(calibrator, dict):
        t = float(calibrator.get("temperature", 1.0))
        t = max(t, 1e-6)
        p = min(max(prob, 1e-6), 1 - 1e-6)
        logit = np.log(p / (1 - p))
        prob = float(1 / (1 + np.exp(-logit / t)))
    elif calibration_type == "sigmoid" and hasattr(calibrator, "predict_proba"):
        p = min(max(prob, 1e-6), 1 - 1e-6)
        logit = np.log(p / (1 - p))
        prob = float(calibrator.predict_proba(np.array([[logit]]))[:, 1][0])
    return {
        "probability": prob,
        "pred": int(prob >= 0.84),
        "risk": risk_level(prob),
    }


def append_history(record: dict) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    hist = PRED_DIR / "history.csv"
    row = pd.DataFrame([record])
    if hist.exists():
        prev = pd.read_csv(hist)
        out = pd.concat([prev, row], axis=0).tail(500)
    else:
        out = row
    out.to_csv(hist, index=False, encoding="utf-8")


st.set_page_config(page_title="智能垃圾邮件过滤器", layout="wide")
st.title("智能垃圾邮件过滤器")

page = st.sidebar.radio(
    "页面",
    ["单封邮件检测", ".eml 文件检测", "批量 CSV 检测", "错误分析", "模型与指标", "历史记录"],
)

if page == "单封邮件检测":
    text = st.text_area("输入邮件正文", height=220)
    if st.button("检测"):
        result = predict_text(text)
        st.write({"是否垃圾": bool(result["pred"]), "概率": result["probability"], "风险": result["risk"]})
        append_history({"mode": "text", "probability": result["probability"], "risk": result["risk"], "text_preview": text[:120]})

elif page == ".eml 文件检测":
    uploaded = st.file_uploader("上传 .eml 文件", type=["eml"])
    if uploaded is not None:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = PRED_DIR / "_tmp.eml"
        tmp.write_bytes(uploaded.getvalue())
        parsed = parse_eml_file(tmp)
        result = predict_text(parsed.clean_text)
        st.write({"主题": parsed.subject, "发件人": parsed.sender, "概率": result["probability"], "风险": result["risk"]})
        append_history({"mode": "eml", "probability": result["probability"], "risk": result["risk"], "text_preview": parsed.clean_text[:120]})

elif page == "批量 CSV 检测":
    uploaded = st.file_uploader("上传 CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        text_col = next((c for c in df.columns if c.lower() in {"text", "body", "clean_text", "message"}), None)
        if text_col:
            probs = [predict_text(str(t))["probability"] for t in df[text_col].fillna("")]
            out = df.copy()
            out["probability"] = probs
            out["risk"] = out["probability"].map(risk_level)
            st.dataframe(out)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("导出 CSV", data=csv_bytes, file_name="spam_detection_results.csv", mime="text/csv")
            append_history({"mode": "csv_batch", "probability": float(out["probability"].mean()), "risk": "batch", "text_preview": f"rows={len(out)}"})

elif page == "错误分析":
    path = ERROR_DIR / "step9_typical_error_cases.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path))
    else:
        st.info("暂无错误分析文件。")

elif page == "模型与指标":
    for p in ["step5_model_summary.csv", "step6_imbalance_strategy_comparison.csv", "step7_calibration_comparison.csv", "step8_threshold_analysis.csv"]:
        f = RESULTS_DIR / p
        if f.exists():
            st.subheader(p)
            st.dataframe(pd.read_csv(f))

elif page == "历史记录":
    hist = PRED_DIR / "history.csv"
    if hist.exists():
        st.dataframe(pd.read_csv(hist))
    if st.button("清空记录") and hist.exists():
        hist.unlink()
        st.rerun()
