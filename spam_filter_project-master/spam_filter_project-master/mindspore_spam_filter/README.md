# MindSpore 智能垃圾邮件过滤器

这是独立于原 `src/` 的 MindSpore 实现，复用项目中的邮件 CSV 数据，交付内容包括：

- MindSpore 二分类模型训练与推理
- 不平衡数据处理：weighted BCE with logits，默认正类权重为 `ham_count / spam_count`
- 概率输出与置信度：`probability_spam`、`confidence`
- 验证集阈值选择：默认要求 `Precision >= 0.95` 后最大化 F1
- 不同特征对比：`structured`、`word_svd`、`char_svd`、`hybrid`
- 错误分析导出：`test_errors.csv`
- 实践报告生成：`outputs/mindspore_practice_report.md`

## 环境

先安装与本机硬件匹配的 MindSpore，再安装其余依赖：

```bash
pip install -r mindspore_spam_filter/requirements_mindspore.txt
```

如果训练时报 Git LFS 指针错误，说明当前 CSV 不是实际数据文件，需要先执行：

```bash
git lfs pull
```

## 训练全部特征并生成报告

在项目根目录运行：

```bash
python mindspore_spam_filter/train.py --feature all --epochs 30 --calibrate
```

当前仓库默认读取：

- `data/splits/train.csv`
- `data/splits/valid.csv`
- `data/splits/test.csv`

也可以显式指定三份 CSV：

```bash
python mindspore_spam_filter/train.py ^
  --train_csv data/splits/train.csv ^
  --valid_csv data/splits/valid.csv ^
  --test_csv data/splits/test.csv ^
  --feature hybrid ^
  --epochs 30 ^
  --calibrate
```

输出目录：

- `mindspore_spam_filter/outputs/feature_comparison.csv`
- `mindspore_spam_filter/outputs/best/`
- `mindspore_spam_filter/outputs/mindspore_practice_report.md`

## 单封邮件预测

```bash
python mindspore_spam_filter/predict.py ^
  --model_dir mindspore_spam_filter/outputs/best ^
  --text "Congratulations, you win a free prize. Click now!"
```

## 批量 CSV 预测

输入 CSV 需要包含 `clean_text`、`text`、`body`、`content`、`message` 或 `email_text` 任一文本列。

```bash
python mindspore_spam_filter/predict.py ^
  --model_dir mindspore_spam_filter/outputs/best ^
  --input_csv data/splits/test.csv ^
  --output_csv mindspore_spam_filter/outputs/batch_predictions.csv
```

## 设计说明

模型主体是 MindSpore `nn.Cell`。`--hidden_dim 0` 时等价于 MindSpore Logistic Regression；默认 `--hidden_dim 128` 使用一层 MLP。文本 TF-IDF 与 SVD 属于特征工程环节，模型训练、损失、checkpoint、推理 logits/probability 均由 MindSpore 完成。

