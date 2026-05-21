# 智能垃圾邮件过滤器课程项目

这是一个可复现的企业邮箱垃圾邮件过滤器课程项目，包含数据下载与清洗、特征工程、模型训练、概率校准、阈值分析、错误分析、Streamlit 原型和完整实验报告。

## 快速上手

### 1. 安装环境

```bash
cd spam_filter_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 一键验收

```bash
python -m src.utils.acceptance_test
```

### 3. 启动应用

```bash
streamlit run app/streamlit_app.py
```

## 项目现状

- 主数据集：5000 封邮件
- 类别比例：ham 4000 / spam 1000
- 划分：train 3500 / valid 750 / test 750
- 最终部署模型：校准后的 Logistic Regression
- 验收状态：当前核心交付已完成

## 目录总览

### 根目录文件

- `README.md`：项目总说明、接手指南、启动方式、文件索引
- `requirements.txt`：Python 依赖
- `pyproject.toml`：项目构建与工具配置

### `configs/`

- `config.yaml`：全局实验配置，包含数据规模、阈值扫描区间、校准策略、路径别名

### `data/`

- `raw/`：原始邮件数据，按语料来源保存
- `cleaned/`：清洗后邮件、去重后邮件、主数据集
- `splits/`：固定训练/验证/测试划分
- `external/`：下载得到的原始压缩包

关键文件：
- `data/raw/`：原始解析结果，适合追溯数据来源
- `data/cleaned/all_parsed_raw.csv`：全部解析邮件的总表
- `data/cleaned/all_parsed_clean_no_corrupt.csv`：去除损坏样本后
- `data/cleaned/all_parsed_clean_dedup.csv`：去重后样本
- `data/cleaned/main_dataset_5000.csv`：最终 5000 封主数据集
- `data/splits/train.csv`：训练集
- `data/splits/valid.csv`：验证集
- `data/splits/test.csv`：测试集
- `data/splits/main_dataset_split.csv`：主数据集分组切分记录

### `papers/`

- `literature_matrix.csv`：9 篇固定文献矩阵
- `literature_download_log.csv`：文献下载日志
- `literature_index_only.csv`：曾经仅保留索引、后来已补齐的记录
- `pdfs/`：9 篇核心论文 PDF
- `summaries/`：Step 3 相关工作总结

关键文件：
- `papers/pdfs/01_sahami_1998_bayesian_filtering_junk_email.pdf`
- `papers/pdfs/02_drucker_1999_svm_spam_categorization.pdf`
- `papers/pdfs/03_androutsopoulos_2000_nb_vs_keyword_antispam.pdf`
- `papers/pdfs/04_metsis_2006_which_naive_bayes.pdf`
- `papers/pdfs/05_guzella_2009_review_ml_spam_filtering.pdf`
- `papers/pdfs/06_platt_1999_probabilistic_outputs_svm.pdf`
- `papers/pdfs/07_niculescu_mizil_caruana_2005_predicting_good_probabilities.pdf`
- `papers/pdfs/08_guo_2017_calibration_modern_neural_networks.pdf`
- `papers/pdfs/09_devlin_2019_bert.pdf`
- `papers/summaries/step3_related_work.md`：文献综述正文

### `src/`

#### `src/data/`
- `email_parser.py`：邮件解析、主题归一化、EML 处理
- `dataset_schema.py`：数据字段定义与结构约束
- `build_dataset.py`：构建主数据集、去重、切分

#### `src/features/`
- `text_features.py`：词/字符 TF-IDF 与结构化特征提取
- `build_features.py`：批量生成特征文件

#### `src/models/`
- 模型相关公共接口与模型封装

#### `src/calibration/`
- `calibration_utils.py`：Platt scaling、sigmoid、temperature scaling 等校准工具

#### `src/evaluation/`
- `metrics.py`：Precision、Recall、F1、Macro-F1、PR-AUC、ROC-AUC、Brier、ECE
- `thresholds.py`：阈值扫描与最优阈值选择
- `run_threshold_analysis.py`：阈值分析脚本
- `generate_figures.py`：生成混淆矩阵、ROC、PR、校准曲线

#### `src/explainability/`
- `error_analysis.py`：错例分类、错误归因、解释性输出
- `run_error_analysis.py`：错误分析脚本

#### `src/training/`
- `training_utils.py`：训练辅助函数
- `train_classical_models.py`：训练 NB / LR / SVM 等经典模型
- `compare_imbalance_strategies.py`：不平衡处理对比
- `run_calibration_experiments.py`：校准实验

#### `src/utils/`
- `io_utils.py`：通用读写工具
- `acceptance_test.py`：全量验收脚本

### `experiments/`

- `logs/`：训练、清洗、校准、阈值分析日志
- `results/`：验证集预测、性能表、阈值扫描表、校准对比表
- `configs/`：实验配置快照

常见结果文件：
- `experiments/results/step5_model_summary.csv`：模型训练汇总
- `experiments/results/step6_imbalance_strategy_comparison.csv`：不平衡策略对比
- `experiments/results/step7_calibration_comparison.csv`：校准前后对比
- `experiments/results/step8_threshold_analysis.csv`：阈值分析
- `experiments/results/*_valid_predictions.csv`：验证集预测结果
- `experiments/results/*_valid_metrics.json`：验证集指标

### `models/`

- `raw/`：未校准模型和向量器
- `calibrated/`：校准后模型
- `best/`：最终部署模型（若后续单独导出）

当前常用文件：
- `models/raw/multinomial_nb.joblib`
- `models/raw/logistic_regression.joblib`
- `models/raw/linear_svm.joblib`
- `models/raw/word_tfidf.joblib`
- `models/raw/char_tfidf.joblib`
- `models/raw/structured_scaler.joblib`
- `models/raw/hybrid_artifacts.joblib`
- `models/calibrated/logistic_regression_calibrated.joblib`
- `models/calibrated/linear_svm_calibrated.joblib`

### `outputs/`

- `figures/`：混淆矩阵、ROC、PR、校准曲线
- `tables/`：统计表、验收报告、特征汇总、错误分类表
- `predictions/`：批量预测结果与导出文件
- `error_analysis/`：典型错例、误判分析、解释性结果

关键文件：
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curve.png`
- `outputs/figures/pr_curve.png`
- `outputs/figures/calibration_curve.png`
- `outputs/tables/acceptance_report.json`
- `outputs/error_analysis/step9_typical_error_cases.csv`
- `outputs/error_analysis/step9_error_analysis_log.json`

### `app/`

- `streamlit_app.py`：应用主入口
- `pages/1_单封邮件检测.py`：单封邮件检测
- `pages/2_eml文件检测.py`：EML 文件检测
- `pages/3_批量CSV检测.py`：批量检测与导出
- `pages/4_错误分析.py`：错误分析展示
- `pages/5_模型与指标.py`：模型对比与曲线展示
- `pages/6_历史记录.py`：历史记录查看与清空
- `assets/`：静态资源

### `tests/`

- `test_metrics.py`：指标计算测试
- `test_thresholds.py`：阈值选择测试
- `test_split_logic.py`：数据划分测试
- `test_calibration.py`：校准测试
- `test_error_analysis.py`：错误分析测试
- `test_step8_threshold_selection.py`：Step 8 阈值选择回归测试

## 常用命令

```bash
python -m src.utils.acceptance_test
python -m src.evaluation.generate_figures
python -m src.evaluation.run_threshold_analysis
python -m src.explainability.run_error_analysis
streamlit run app/streamlit_app.py
```

## 说明

- 所有路径均使用项目相对路径
- 所有实验结果都已落盘，优先查看 `experiments/results/`、`outputs/`、`reports/`
- 若需要复现实验，建议按 Step 2 -> Step 11 的顺序重跑
