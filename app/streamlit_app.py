import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data.email_parser import parse_eml_file
from src.evaluation.mindspore_metrics import sigmoid_np
from src.explainability.error_analysis import classify_error_case
from src.models.mindspore_model import build_cells, require_mindspore
from src.utils.io_utils import load_yaml

ROOT = Path(project_root)

CFG = load_yaml(ROOT / "configs" / "config.yaml")
RESULTS_DIR = ROOT / "experiments" / "results"
ERROR_DIR = ROOT / "outputs" / "error_analysis"
TABLES_DIR = ROOT / "outputs" / "tables"
MODELS_DIR = ROOT / "outputs" / "mindspore" / "best"
PRED_DIR = ROOT / "outputs" / "predictions"


@st.cache_resource
def load_default_model():
    metadata_path = MODELS_DIR / "metadata.json"
    bundle_path = MODELS_DIR / "feature_bundle.joblib"
    checkpoint_path = MODELS_DIR / "model.ckpt"
    if not (metadata_path.exists() and bundle_path.exists() and checkpoint_path.exists()):
        return None

    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ms, _, _, _ = require_mindspore()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    Network, _ = build_cells(int(metadata["input_dim"]), int(metadata["hidden_dim"]), float(metadata["dropout"]))
    network = Network()
    params = ms.load_checkpoint(str(checkpoint_path))
    ms.load_param_into_net(network, params)
    network.set_train(False)
    return {"metadata": metadata, "bundle": joblib.load(bundle_path), "network": network}


def risk_level(prob: float) -> str:
    if prob < 0.30:
        return "安全"
    if prob < 0.70:
        return "可疑"
    return "垃圾"


def predict_text(text: str) -> dict:
    model_bundle = load_default_model()
    if model_bundle is None:
        return {"probability": 0.0, "label": "不可用"}
    ms, Tensor, _, _ = require_mindspore()
    features = model_bundle["bundle"].transform([text])
    logits = model_bundle["network"](Tensor(features.astype(np.float32), ms.float32)).asnumpy()
    temperature = float(model_bundle["metadata"].get("temperature", 1.0))
    threshold = float(model_bundle["metadata"].get("threshold", 0.5))
    prob = float(sigmoid_np(logits / temperature)[0])
    return {
        "probability": prob,
        "pred": int(prob >= threshold),
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
