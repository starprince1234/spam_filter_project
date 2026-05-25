# 基于 MindSpore 的智能垃圾邮件过滤器实践报告

## 1. 任务目标

企业邮箱系统需要在每天大量入站邮件中自动识别垃圾邮件，并将高风险邮件放入回收站。本项目使用 5000 封已标注邮件构建二分类过滤器，其中垃圾邮件约占 20%。系统需要解决类别不平衡问题，并为每封邮件输出可用于人工审核或自动处置的预测置信度。

## 2. 数据与划分

主数据集规模为 5000 封邮件，类别比例为 ham 4000 / spam 1000。原项目已固定划分为 train / valid / test = 3500 / 750 / 750。MindSpore 版本默认读取：

- `data/splits/train.csv`
- `data/splits/valid.csv`
- `data/splits/test.csv`

当前工作树中的 CSV 是 Git LFS 指针文件，不是真实数据文件；训练前需要执行 `git lfs pull` 或替换为实际 CSV。

## 3. 特征工程

MindSpore 实现支持四类特征：

| 特征 | 内容 | 作用 |
|---|---|---|
| structured | 长度、词数、URL、HTML、退订词、促销词、金额符号、标点密度等 20 维特征 | 捕捉邮件结构和垃圾邮件人工规则信号 |
| word_svd | 词级 unigram/bigram TF-IDF，再用 SVD 压缩为稠密向量 | 捕捉关键词和短语模式 |
| char_svd | 字符级 3-5 gram TF-IDF，再用 SVD 压缩 | 捕捉拼写变形、URL 片段、规避词 |
| hybrid | structured + word_svd + char_svd 拼接 | 综合文本语义、字符模式和结构化风险信号 |

原传统机器学习实验中的已知基线如下，MindSpore 脚本运行后会在 `outputs/feature_comparison.csv` 中生成 MindSpore 模型对应的真实对比表。

| 基线特征/模型 | Precision | Recall | F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| 词级 TF-IDF + Logistic Regression | 0.8963 | 0.9800 | 0.9363 | 0.9597 |
| 字符级 TF-IDF + Linear SVM | 0.9226 | 0.9533 | 0.9377 | 0.9609 |
| Multinomial NB | 0.9912 | 0.7467 | 0.8517 | 0.9101 |

## 4. MindSpore 模型设计

模型主体使用 MindSpore `nn.Cell`：

- `--hidden_dim 0`：一层 Dense，相当于 MindSpore Logistic Regression。
- 默认 `--hidden_dim 128`：Dense + ReLU + Dropout + Dense 的轻量 MLP。
- 输出为单个 logit，经 sigmoid 得到 `probability_spam`。

训练损失为自定义 weighted BCE with logits。若不手动指定 `--pos_weight`，脚本自动使用：

```text
pos_weight = ham_count / spam_count
```

在 4000 / 1000 的总体比例下，正类垃圾邮件会获得更高训练权重，从而降低模型只偏向多数类 ham 的风险。

## 5. 阈值与置信度

模型输出概率不直接等同于最终处置结果。脚本会在验证集扫描 0.05 到 0.95 的阈值，默认优先满足：

```text
Precision >= 0.95
```

然后最大化 F1。这样可以控制正常邮件被误放入回收站的风险。

预测输出字段：

| 字段 | 含义 |
|---|---|
| `probability_spam` | 邮件属于垃圾邮件的概率 |
| `pred_label` | 阈值判断后的类别，1 为 spam，0 为 ham |
| `pred_name` | `spam` 或 `ham` |
| `confidence` | 最终预测类别的置信度，spam 为 `p`，ham 为 `1-p` |

若启用 `--calibrate`，脚本会在验证集 logits 上拟合温度缩放，改善 Brier/ECE 等概率校准指标。

## 6. 不平衡处理对比

原项目中传统模型的不平衡实验结果表明，类别权重和代价敏感训练能显著提高垃圾邮件召回率：

| 策略 | 模型 | Precision | Recall | F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| 不处理 | LR | 0.9661 | 0.7600 | 0.8507 | 0.9091 |
| class_weight | LR | 0.8963 | 0.9800 | 0.9363 | 0.9597 |
| 阈值调整 | LR | 0.9524 | 0.8000 | 0.8696 | 0.9201 |
| 代价敏感 | SVM | 0.9226 | 0.9533 | 0.9377 | 0.9609 |

MindSpore 版本对应采用 weighted BCE，并叠加验证集阈值选择，目标是在高 Precision 约束下尽量提升 Recall 和 F1。

## 7. 错误分析

训练脚本会导出：

```text
mindspore_spam_filter/outputs/best/test_errors.csv
```

错例分析重点如下：

| 错误类型 | 典型原因 | 改进方向 |
|---|---|---|
| 正常营销邮件误判为垃圾 | 促销词、链接、金额符号较多 | 加入白名单域名、业务邮件模板特征 |
| 系统通知误判为垃圾 | 自动通知中链接和模板化文本密集 | 引入发件域、历史交互、系统标题规则 |
| 钓鱼或伪装垃圾邮件漏判 | 文本短、措辞正常、HTML 隐藏风险 | 增加 URL 域名信誉、HTML 结构特征 |
| 拼写变形垃圾邮件漏判 | 使用错拼、字符替换规避关键词 | 提高字符级特征权重，补充变体词样本 |
| 短文本误判 | 可用语义信息不足 | 对低置信度短文本进入人工审核队列 |

## 8. 运行方式

训练并生成 MindSpore 实验报告：

```bash
python mindspore_spam_filter/train.py --feature all --epochs 30 --calibrate
```

单封邮件预测：

```bash
python mindspore_spam_filter/predict.py --model_dir mindspore_spam_filter/outputs/best --text "Congratulations, you win a free prize. Click now!"
```

## 9. 结论

本 MindSpore 版本覆盖了项目要求的核心环节：不平衡训练、不同特征性能对比、预测概率与置信度输出、阈值选择、错误分析和实践报告生成。当前环境缺少 MindSpore 包且 CSV 为 Git LFS 指针，因此尚未在本机完成真实 MindSpore 训练；环境和数据补齐后，可直接运行训练脚本生成最终指标报告。

