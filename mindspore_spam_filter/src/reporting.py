from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda v: f"{v:.4f}")
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in text.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_practice_report(
    output_path: Path,
    comparison_csv: Path,
    best_feature: str,
    threshold: float,
    metrics: dict,
    error_csv: Path | None = None,
) -> None:
    comparison = pd.read_csv(comparison_csv)
    lines = [
        "# MindSpore 智能垃圾邮件过滤器实践报告",
        "",
        "## 1. 任务目标",
        "",
        "本实现面向企业邮箱垃圾邮件识别场景，使用 MindSpore 训练二分类模型，输出垃圾邮件概率和最终预测置信度。",
        "数据不平衡比例按 4000 ham / 1000 spam 处理，训练阶段采用正类加权 BCE，部署阶段在验证集上选择阈值。",
        "",
        "## 2. 特征方案",
        "",
        "- structured：长度、标点、URL、HTML、退订词、促销词等 20 维人工特征。",
        "- word_svd：词级 TF-IDF 后用 TruncatedSVD 压缩为稠密向量。",
        "- char_svd：字符级 3-5 gram TF-IDF 后用 TruncatedSVD 压缩。",
        "- hybrid：结构化特征、词级 SVD、字符级 SVD 拼接。",
        "",
        "## 3. 不平衡处理与置信度",
        "",
        "- 损失函数：MindSpore 自定义 weighted BCE with logits，正类权重默认为 negative / positive。",
        "- 阈值：在验证集扫描 0.05 到 0.95，优先满足 Precision 下限，再最大化 F1。",
        "- 置信度：模型输出 sigmoid 概率；若启用温度校准，先用验证集拟合温度，再输出校准后的 probability_spam。",
        "",
        "## 4. 不同特征下的性能对比",
        "",
        markdown_table(comparison),
        "",
        "## 5. 最终模型",
        "",
        f"- 最优特征：{best_feature}",
        f"- 部署阈值：{threshold:.4f}",
        f"- 测试集 Precision：{metrics.get('precision', float('nan')):.4f}",
        f"- 测试集 Recall：{metrics.get('recall', float('nan')):.4f}",
        f"- 测试集 F1：{metrics.get('f1', float('nan')):.4f}",
        f"- 测试集 Macro-F1：{metrics.get('macro_f1', float('nan')):.4f}",
        f"- 测试集 PR-AUC：{metrics.get('pr_auc', float('nan')):.4f}",
        f"- 测试集 ROC-AUC：{metrics.get('roc_auc', float('nan')):.4f}",
        f"- 测试集 Brier：{metrics.get('brier', float('nan')):.4f}",
        f"- 测试集 ECE：{metrics.get('ece', float('nan')):.4f}",
        "",
        "## 6. 错误分析",
        "",
        "脚本会导出测试集错例 CSV，用于人工复核。重点关注以下错误来源：",
        "",
        "- 正常营销或系统通知被误判为垃圾邮件：链接、金额、促销词较密集。",
        "- 垃圾邮件漏判：短文本、伪装正常业务措辞、HTML/URL 特征不明显。",
        "- 置信度高但错误的样本：应优先进入人工审核队列，用于后续增量标注。",
    ]
    if error_csv:
        lines.extend(["", f"错例文件：`{error_csv}`"])
    lines.extend(["", "## 7. 结论", "", "MindSpore 版本已经覆盖特征对比、不平衡训练、概率校准、阈值选择、置信度输出和错例导出，可作为课程交付代码基础。"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
