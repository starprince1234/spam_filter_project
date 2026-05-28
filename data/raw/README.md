# Raw Data Policy

这里保留的是原始语料的入口说明。

- `data/external/` 保存可复现的原始压缩包
- `data/cleaned/main_dataset_5000.csv` 保存最终主数据集
- `data/splits/` 保存训练/验证/测试切分
- `data/raw/**/*.txt` 是由原始语料展开后的中间邮件文件，体量很大，已通过 `.gitignore` 排除

如需重建中间 raw 文本，可从 `data/external/` 的原始压缩包重新解包并运行 `src/data/build_dataset.py`。
