# MindSpore 智能垃圾邮件过滤器

> 基于国产自研 AI 框架 **MindSpore 2.6.0（CPU / PyNative 模式）** 构建的可复现企业级邮箱垃圾邮件二分类系统。
>
> 项目目标：在公开邮件语料（Enron-Spam、SpamAssassin、TREC）上完成数据治理 → 特征工程 → 分类训练 → 概率校准 → 阈值优化 → 错误分析 → 工程化交付的完整闭环，并通过 Streamlit 应用对外提供可视化检测能力。

---

## 1. 项目简介与背景

### 1.1 学术与现实意义

垃圾邮件（Spam）每天占据全球邮件流量的 45% 以上，给企业邮箱带来**生产力损失**、**钓鱼攻击**、**勒索软件投递**等多重风险。一个合格的垃圾邮件过滤器需要同时满足三件事：

1. **高精度**：误把正常邮件判为垃圾邮件（False Positive）会直接造成业务损失，所以工业系统通常对 Precision 设硬下限（如 ≥ 0.95）。
2. **可校准的概率**：下游需要根据“概率分数”进行风险分级、二次审核或人工抽查，因此模型输出必须接近真实概率，而不仅仅是“是否大于 0.5”。
3. **可解释 + 可复现**：可以解释为什么一封邮件被判为垃圾，并能在新的数据集上重跑实验，得到可对比的结果。

### 1.2 技术选型

- **框架**：MindSpore 2.6.0 CPU 版（Windows wheel，Python 3.9）。
- **执行模式**：PyNative 动态图，便于课程作业调试。
- **网络结构**：`nn.Cell` 实现的线性 / 单隐层分类器（`hidden_dim=0` 时退化为 Logistic Regression）。
- **损失函数**：手写的 weighted Binary Cross Entropy with logits，支持类别不平衡补偿。
- **训练接口**：`mindspore.value_and_grad` + `nn.Adam`，数据使用 `mindspore.dataset.GeneratorDataset` 批处理。
- **特征**：scikit-learn TF-IDF + 自研 20 维结构化特征 + 可选 TruncatedSVD 压缩。
- **校准**：温度缩放（Temperature Scaling）。
- **阈值选择**：在验证集上扫描 0.05–0.95，约束 Precision ≥ 0.95，再最大化 F1。

> 项目代码完全去除了 PyTorch 依赖。除少量 sklearn 工具（特征工程、指标）外，训练侧完全使用 MindSpore 原生 API。

---

## 2. 严谨的实验设计与文献支撑

### 2.1 核心论文矩阵（[`papers/literature_matrix.csv`](papers/literature_matrix.csv)）

| 编号 | 论文 | 年份 | 核心方法 | 在本项目中的作用 |
|---|---|---|---|---|
| 1 | Sahami et al. *A Bayesian Approach to Filtering Junk E-Mail* | 1998 | Naive Bayes 概率过滤 | 概率式文本分类的理论基线，奠定 NB / 词袋特征的对照实验 |
| 2 | Drucker, Wu, Vapnik *SVM for Spam Categorization* | 1999 | SVM 文本分类 | 提供 Linear SVM 基线，启发 margin 视角与校准必要性 |
| 3 | Androutsopoulos et al. *NB vs Keyword Anti-Spam* | 2000 | 早期对比实验 | 实验设计的范式参考，强调可重复对比 |
| 4 | Metsis et al. *Which Naive Bayes?* | 2006 | NB 各变体对比 | 帮助选择 MultinomialNB 作为 NB 基线 |
| 5 | Guzella & Caminhas *Survey of ML for Spam Filtering* | 2009 | 综述 | 经典 vs 现代方法的分类学，写作框架来源 |
| 6 | Platt *Probabilistic Outputs for SVM* | 1999 | Platt scaling | SVM 概率化与 sigmoid 校准的理论依据 |
| 7 | Niculescu-Mizil & Caruana *Predicting Good Probabilities* | 2005 | 校准评估 | Brier、可靠性图、ECE 指标的依据 |
| 8 | Guo et al. *On Calibration of Modern Neural Networks* | 2017 | Temperature Scaling | 本项目温度校准实现的直接依据 |
| 9 | Devlin et al. *BERT* | 2019 | 预训练 Transformer | 现代深度文本基线，作为对照参考 |

> 完整 PDF 与下载日志见 [`papers/`](papers/) 目录，文献综述见 [`papers/summaries/step3_related_work.md`](papers/summaries/step3_related_work.md)。

### 2.2 数据集

主数据集固定为 5000 封邮件，类别比例 ham:spam = 4000:1000，划分 train:valid:test = 3500:750:750。来源包括：

- Enron-Spam Dataset
- SpamAssassin Public Corpus
- TREC Spam Track（含官方 index 与 trec07p 预处理 CSV）

数据治理脚本：[`src/data/build_dataset.py`](src/data/build_dataset.py)、邮件解析：[`src/data/email_parser.py`](src/data/email_parser.py)、最终数据：[`data/cleaned/main_dataset_5000.csv`](data/cleaned/main_dataset_5000.csv)。

---

## 3. 四套特征方案消融设计

所有方案的实现位于 [`src/features/mindspore_features.py`](src/features/mindspore_features.py)，支持以 `--feature` 参数切换：

### 方案 A：词级 TF-IDF（`--feature word`）

- `TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=30000)`
- 输出维度 = 30000，捕获 unigram + bigram 词共现。
- 适合长文本邮件，特征解释性高。

### 方案 B：字符级 TF-IDF（`--feature char`）

- `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000)`
- 对错别字、变形拼写、URL 片段稳健，能够捕捉“pr1ze / w1n”这类反垃圾对抗写法。

### 方案 C：20 维结构化特征（`--feature structured`）

20 个手工特征：字符长度、词数、句数、感叹号 / 问号数、大写字母比例、数字比例、URL / Email / HTML 标签计数、HTML 标志、附件提示、退订关键词、$ 符号、电话号码模式、标点密度、垃圾词典命中、促销词典命中、过多链接标志、重复字符标志。经过 `StandardScaler`。

### 方案 D：混合拼接特征（`--feature hybrid` / `--feature word_svd` / `--feature char_svd` / `--feature hybrid_svd`）

- `hybrid` = 词级 TF-IDF ⊕ 字符级 TF-IDF ⊕ 结构化特征（稀疏拼接）。
- `*_svd` = TF-IDF 经过 TruncatedSVD（默认 200 维）压缩后再拼接结构化特征，得到稠密低维向量，便于 MindSpore 线性层稳定训练。

> 默认 `--feature word` 是与原参考项目 Logistic Regression 实验对齐的基线；推荐在工程交付时使用 `word`，在轻量化部署时考虑 `word_svd`。

---

## 4. 完整项目目录树深度解析

```
spam_filter_project_mindspore/
├── README.md                       本文件，交付级说明
├── requirements.txt                Python 依赖清单（含 mindspore==2.6.0）
├── pyproject.toml                  setuptools 元信息（package-dir = src）
├── configs/
│   └── config.yaml                 全局实验配置（路径、阈值、校准策略）
├── data/
│   ├── external/                   原始压缩包（SpamAssassin / Enron-Spam）
│   ├── raw/                        逐封邮件解析的原文文本
│   ├── cleaned/
│   │   └── main_dataset_5000.csv   清洗、去重后的 5000 封主数据集
│   └── splits/
│       ├── train.csv               训练集 3500 封
│       ├── valid.csv               验证集 750 封
│       ├── test.csv                测试集 750 封
│       └── main_dataset_split.csv  全量切分审计表
├── src/                            主源码（已彻底去 torch 化，全部 MindSpore）
│   ├── data/
│   │   ├── email_parser.py         .eml 解析、HTML 抽取、主题标准化
│   │   ├── dataset_schema.py       邮件记录的 dataclass schema
│   │   ├── build_dataset.py        语料下载、解析、去重、切分
│   │   └── mindspore_dataset.py    CSV 加载、标签归一化、GeneratorDataset
│   ├── features/
│   │   ├── text_features.py        scikit-learn 特征工具（保留作经典对照）
│   │   ├── build_features.py       批量生成与缓存特征矩阵
│   │   └── mindspore_features.py   ★ 训练用特征 bundle（A/B/C/D 方案）
│   ├── models/
│   │   └── mindspore_model.py      ★ nn.Cell 分类器、weighted BCE Loss、predict_logits
│   ├── calibration/
│   │   └── calibration_utils.py    sigmoid / temperature 校准工具
│   ├── evaluation/
│   │   ├── metrics.py              经典指标
│   │   ├── thresholds.py           阈值扫描
│   │   ├── generate_figures.py     ROC / PR / 混淆矩阵 / 校准曲线
│   │   ├── run_threshold_analysis.py
│   │   ├── mindspore_metrics.py    ★ MindSpore 端指标 + 阈值 + 温度校准
│   │   └── evaluate_mindspore.py   ★ 加载 best 模型在 valid+test 上完整评估
│   ├── training/
│   │   ├── training_utils.py       通用训练辅助
│   │   ├── train_classical_models.py            经典 ML 训练（NB / LR / SVM）
│   │   ├── compare_imbalance_strategies.py      不平衡策略对比
│   │   ├── run_calibration_experiments.py       概率校准实验
│   │   └── train.py                ★ MindSpore 训练主入口（PyNative + value_and_grad）
│   ├── prediction/
│   │   └── predict.py              ★ 单文本 / CSV 推理 CLI
│   ├── explainability/
│   │   ├── error_analysis.py       错例归因与典型错例导出
│   │   └── run_error_analysis.py
│   └── utils/
│       ├── io_utils.py             读写工具
│       └── acceptance_test.py      项目级一键验收
├── app/                            Streamlit 多页面前端
│   ├── streamlit_app.py            主入口，加载 outputs/mindspore/best
│   └── pages/
│       ├── 1_单封邮件检测.py
│       ├── 2_eml文件检测.py
│       ├── 3_批量CSV检测.py
│       ├── 4_错误分析.py
│       ├── 5_模型与指标.py
│       └── 6_历史记录.py
├── experiments/                    经典 ML 实验产出
│   ├── logs/                       实验日志
│   └── results/                    模型对比、校准、阈值表
├── outputs/
│   ├── features/                   缓存的稀疏 / 稠密特征矩阵
│   ├── figures/                    ROC、PR、混淆矩阵、校准曲线
│   ├── tables/                     步骤化结果表
│   ├── error_analysis/             典型错例 / 错误归因 CSV
│   └── mindspore/
│       ├── word/                   --feature word 的 10 epoch 训练产物
│       └── best/                   ★ 最终交付权重（model.ckpt + metadata.json + ...）
├── papers/                         9 篇核心文献 PDF + 矩阵 + 综述
├── reports/                        最终报告与交付清单
└── tests/                          指标 / 阈值 / 校准 / 错例分析的单元测试
```

> 每个标 ★ 的文件是 MindSpore 版核心，请在交付时优先 review。

---

## 5. 复现与部署指南

### 5.1 环境配置（Windows，已在 Python 3.9.25 验证）

```bash
conda create -y -n mindspore_env python=3.9
conda activate mindspore_env

pip install mindspore==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import mindspore; mindspore.run_check()"
```

预期输出：

```
MindSpore version:  2.6.0
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

若报错：

```
# 强制降级 protobuf 到兼容版本
pip install protobuf==3.20.3 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 训练（10 epoch 正式运行）

```bash
python -m src.training.train --epochs 10 --feature word --hidden_dim 0
```

可调超参（节选）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--feature` | `word` | `word` / `char` / `structured` / `hybrid` / `word_svd` / `char_svd` / `hybrid_svd` |
| `--epochs` | 30 | 训练轮数 |
| `--hidden_dim` | 0 | 0 时为线性分类器，>0 时为单隐层 MLP |
| `--learning_rate` | 1e-3 | Adam 学习率 |
| `--batch_size` | 128 | 训练批大小 |
| `--precision_floor` | 0.95 | 阈值选择时的精度下限 |
| `--no_calibrate` | off | 关闭温度校准 |

输出：

- [`outputs/mindspore/word/`](outputs/mindspore/word/)：当次特征运行的全部产物
- [`outputs/mindspore/best/`](outputs/mindspore/best/)：最新一次训练复制的最佳权重

### 5.3 评估

```bash
python -m src.evaluation.evaluate_mindspore --model_dir outputs/mindspore/best
```

### 5.4 推理

```bash
# 单封邮件
python -m src.prediction.predict --text "Congratulations, you win a free prize. Click now!"

# 批量 CSV
python -m src.prediction.predict --input_csv data/splits/test.csv --output_csv outputs/mindspore/predictions.csv
```

输入 CSV 必须包含 `clean_text` / `text` / `body` / `content` / `message` / `email_text` 任一文本列。

### 5.5 Streamlit 可视化

```bash
conda activate mindspore_env
streamlit run app/streamlit_app.py
```

默认监听 `http://localhost:8501`。**任何机器克隆仓库后都能一键启动**，因为 [`app/streamlit_app.py`](app/streamlit_app.py) 与 6 个子页面的最顶部都内置了 `sys.path` 动态注入：

```python
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

子页面（多一层 `dirname`）：

```python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

#### 6 个页面功能

| 页面 | 文件 | 功能 |
|---|---|---|
| 单封邮件检测 | `app/pages/1_单封邮件检测.py` | 文本输入 → spam 概率 + 风险等级（安全 / 可疑 / 垃圾） |
| .eml 文件检测 | `app/pages/2_eml文件检测.py` | 上传 .eml → 解析正文 + 头信息 → 分类预测 |
| 批量 CSV 检测 | `app/pages/3_批量CSV检测.py` | 上传 CSV → 自动识别文本列 → 全量打分 + 导出 |
| 错误分析 | `app/pages/4_错误分析.py` | 展示 [`outputs/error_analysis/step9_typical_error_cases.csv`](outputs/error_analysis/step9_typical_error_cases.csv) |
| 模型与指标 | `app/pages/5_模型与指标.py` | 展示 step5 / step6 / step7 / step8 等结果表 |
| 历史记录 | `app/pages/6_历史记录.py` | 显示并清空 `outputs/predictions/history.csv` |

#### Troubleshooting

- **`ModuleNotFoundError: No module named 'src'` / `'app'`**
  优先确认你是在仓库根目录执行 `streamlit run app/streamlit_app.py`。本仓库已通过 `sys.path.insert` 自动注入根目录，理论上无需手动配置。如果遇到 IDE 集成的问题，可在终端临时设置：

  ```cmd
  set PYTHONPATH=.
  streamlit run app/streamlit_app.py
  ```

  PowerShell：

  ```powershell
  $env:PYTHONPATH="."
  streamlit run app/streamlit_app.py
  ```

  Bash：

  ```bash
  export PYTHONPATH=.
  streamlit run app/streamlit_app.py
  ```

- **首次启动时 Streamlit 提示输入邮箱**
  直接回车即可跳过，或在用户目录创建 `~/.streamlit/credentials.toml` 写入：

  ```toml
  [general]
  email = ""
  ```

- **PermissionError: best/model.ckpt**
  代表已有一个进程仍在占用 `outputs/mindspore/best/model.ckpt`（通常是上一次训练 / Streamlit 进程没退干净）。本项目训练脚本已使用 “staging 目录 + 重试 + 清除只读位” 的方式自动恢复；仍失败时手动删除 `outputs/mindspore/best/` 后重跑。

- **MindSpore CPU 版安装失败**
  本项目锁定 `mindspore==2.6.0` 的 cp39 wheel，请务必使用 Python 3.9（本仓库 Conda env 名 `mindspore_env`）。Python 3.10+ 在清华镜像源中没有 2.6.0 wheel。

---

## 6. 最终实验结果与性能榜单

### 6.1 MindSpore 10-epoch 训练（`--feature word --hidden_dim 0`）

环境：Windows 11 / CPU / Python 3.9.25 / MindSpore 2.6.0 PyNative。

训练损失曲线（来自 [`outputs/mindspore/best/training_history.csv`](outputs/mindspore/best/training_history.csv)）：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `train_loss` | 1.0888 | 1.0321 | 0.9810 | 0.9335 | 0.8893 | 0.8482 | 0.8100 | 0.7744 | 0.7413 | 0.7103 |

模型与训练超参：

- `input_dim = 30000`（词级 TF-IDF 1-2 gram）
- `hidden_dim = 0`（等价于 Logistic Regression）
- `pos_weight = 4.0`（按 train 中 ham:spam ≈ 4:1 自动估计）
- `temperature = 0.5`（基于 valid logits 拟合）
- **`threshold = 0.61`**（valid 上扫描 0.05–0.95 后，在 Precision ≥ 0.95 的可行域内最大化 F1）

最终指标（来自 [`outputs/mindspore/best/metadata.json`](outputs/mindspore/best/metadata.json)）：

| 数据集 | Precision | Recall | F1 | Macro-F1 | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| valid | 0.9524 | 0.8000 | 0.8696 | 0.9201 | 0.9782 | 0.9949 | 0.0724 | 0.2085 |
| test | 0.9542 | 0.8333 | 0.8897 | 0.9321 | 0.9768 | 0.9947 | 0.0744 | 0.2045 |

混淆矩阵（test，阈值 0.61）：

|  | 预测 ham | 预测 spam |
|---|---|---|
| 真实 ham | 594 | 6 |
| 真实 spam | 25 | 125 |

> 解读：在 750 封测试邮件上，仅 **6 封** 正常邮件被误判为垃圾邮件（FP = 6），满足 “Precision ≥ 0.95” 的工业要求；同时召回了 **125/150 = 83.3%** 的真实垃圾邮件。

### 6.2 不平衡处理策略对比（来自经典 ML 实验，[`experiments/results/step6_imbalance_strategy_comparison.csv`](experiments/results/step6_imbalance_strategy_comparison.csv)）

| 策略 | 模型 | threshold | Precision | Recall | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|
| none | logistic_regression | 0.50 | 0.9661 | 0.7600 | 0.8507 | 0.0467 | 0.1206 |
| none | linear_svm | 0.50 | 0.9226 | 0.9533 | 0.9377 | 0.0167 | 0.0128 |
| class_weight | logistic_regression | 0.50 | 0.8963 | 0.9800 | 0.9363 | 0.0462 | 0.1332 |
| class_weight | linear_svm | 0.50 | 0.9226 | 0.9533 | 0.9377 | 0.0164 | 0.0117 |
| **threshold_adjust** | **logistic_regression** | **0.70** | **0.9524** | **0.8000** | **0.8696** | 0.0462 | 0.1332 |
| **threshold_adjust** | **linear_svm** | **0.66** | **0.9524** | **0.9333** | **0.9428** | 0.0164 | 0.0117 |
| cost_sensitive | logistic_regression | 0.50 | 0.8963 | 0.9800 | 0.9363 | 0.0462 | 0.1332 |
| cost_sensitive | linear_svm | 0.50 | 0.9226 | 0.9533 | 0.9377 | 0.0164 | 0.0117 |

### 6.3 校准前后指标（[`experiments/results/step7_calibration_comparison.csv`](experiments/results/step7_calibration_comparison.csv)）

| 模型 | 校准方式 | Brier 前 | Brier 后 | ECE 前 | ECE 后 | PR-AUC（前/后） |
|---|---|---|---|---|---|---|
| logistic_regression | temperature | 0.0462 | 0.0263 | 0.1332 | **0.0435** | 0.9757 / 0.9757 |
| linear_svm | sigmoid | 0.0164 | 0.0164 | 0.0117 | **0.0112** | 0.9866 / 0.9866 |

> Logistic Regression 经温度缩放后 ECE 由 0.1332 降至 0.0435，**降幅 67.4%**，校准效果显著。

### 6.4 Precision ≥ 0.95 下的最优阈值（[`experiments/results/step8_threshold_analysis.csv`](experiments/results/step8_threshold_analysis.csv)）

| 模型 | 最优阈值 | 对应 Precision | 对应 Recall | 对应 F1 |
|---|---|---|---|---|
| logistic_regression | 0.84 | 0.9531 | 0.8133 | 0.8777 |
| linear_svm | 0.59 | 0.9524 | 0.9333 | 0.9428 |
| **mindspore-word (本项目主交付)** | **0.61** | **0.9524 / 0.9542** (valid/test) | 0.8000 / 0.8333 | 0.8696 / 0.8897 |

---

## 7. 主要文件速查

| 文件 | 作用 |
|---|---|
| [`src/data/mindspore_dataset.py`](src/data/mindspore_dataset.py) | CSV 加载、标签归一化、`GeneratorDataset` |
| [`src/features/mindspore_features.py`](src/features/mindspore_features.py) | 词 / 字符 TF-IDF、20 维结构化、SVD |
| [`src/models/mindspore_model.py`](src/models/mindspore_model.py) | `nn.Cell` 分类器 + weighted BCE |
| [`src/evaluation/mindspore_metrics.py`](src/evaluation/mindspore_metrics.py) | 指标 + 温度校准 + 阈值扫描 |
| [`src/training/train.py`](src/training/train.py) | MindSpore 训练入口 |
| [`src/evaluation/evaluate_mindspore.py`](src/evaluation/evaluate_mindspore.py) | MindSpore 评估入口 |
| [`src/prediction/predict.py`](src/prediction/predict.py) | 单文本 / CSV 推理 |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Streamlit 主入口 |
| [`outputs/mindspore/best/metadata.json`](outputs/mindspore/best/metadata.json) | 最终交付指标与阈值 |

---

## 8. 致谢

- 数据：Enron-Spam Dataset、SpamAssassin Public Corpus、TREC Spam Track。
- 框架：[MindSpore](https://www.mindspore.cn/)、scikit-learn、Streamlit。
- 文献：见第 2.1 节文献矩阵。
