# MindSpore 智能垃圾邮件过滤器实践报告

## 1. 任务目标

本实现面向企业邮箱垃圾邮件识别场景，使用 MindSpore 训练二分类模型，输出垃圾邮件概率和最终预测置信度。
数据不平衡比例按 4000 ham / 1000 spam 处理，训练阶段采用正类加权 BCE，部署阶段在验证集上选择阈值。

## 2. 特征方案

- structured：长度、标点、URL、HTML、退订词、促销词等 20 维人工特征。
- word_svd：词级 TF-IDF 后用 TruncatedSVD 压缩为稠密向量。
- char_svd：字符级 3-5 gram TF-IDF 后用 TruncatedSVD 压缩。
- hybrid：结构化特征、词级 SVD、字符级 SVD 拼接。

## 3. 不平衡处理与置信度

- 损失函数：MindSpore 自定义 weighted BCE with logits，正类权重默认为 negative / positive。
- 阈值：在验证集扫描 0.05 到 0.95，优先满足 Precision 下限，再最大化 F1。
- 置信度：模型输出 sigmoid 概率；若启用温度校准，先用验证集拟合温度，再输出校准后的 probability_spam。

## 4. 不同特征下的性能对比

| feature | input_dim | threshold | temperature | precision | recall | f1 | macro_f1 | pr_auc | roc_auc | brier | ece | model_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | 420 | 0.5800 | 1.5538 | 0.9664 | 0.9600 | 0.9632 | 0.9770 | 0.9925 | 0.9979 | 0.0126 | 0.0063 | E:\川大上课\模式识别\spam_filter_project-master\spam_filter_project-master\mindspore_spam_filter\outputs\hybrid |
| word_svd | 200 | 0.8400 | 1.4480 | 0.9577 | 0.9067 | 0.9315 | 0.9575 | 0.9844 | 0.9959 | 0.0236 | 0.0104 | E:\川大上课\模式识别\spam_filter_project-master\spam_filter_project-master\mindspore_spam_filter\outputs\word_svd |
| char_svd | 200 | 0.8200 | 1.3631 | 0.9384 | 0.9133 | 0.9257 | 0.9537 | 0.9865 | 0.9966 | 0.0206 | 0.0081 | E:\川大上课\模式识别\spam_filter_project-master\spam_filter_project-master\mindspore_spam_filter\outputs\char_svd |
| structured | 20 | 0.9200 | 0.6890 | 0.8947 | 0.5667 | 0.6939 | 0.8171 | 0.8311 | 0.9228 | 0.0991 | 0.0252 | E:\川大上课\模式识别\spam_filter_project-master\spam_filter_project-master\mindspore_spam_filter\outputs\structured |

## 5. 最终模型

- 最优特征：hybrid
- 部署阈值：0.5800
- 测试集 Precision：0.9664
- 测试集 Recall：0.9600
- 测试集 F1：0.9632
- 测试集 Macro-F1：0.9770
- 测试集 PR-AUC：0.9925
- 测试集 ROC-AUC：0.9979
- 测试集 Brier：0.0126
- 测试集 ECE：0.0063

## 6. 错误分析

脚本会导出测试集错例 CSV，用于人工复核。重点关注以下错误来源：

- 正常营销或系统通知被误判为垃圾邮件：链接、金额、促销词较密集。
- 垃圾邮件漏判：短文本、伪装正常业务措辞、HTML/URL 特征不明显。
- 置信度高但错误的样本：应优先进入人工审核队列，用于后续增量标注。

错例文件：`E:\川大上课\模式识别\spam_filter_project-master\spam_filter_project-master\mindspore_spam_filter\outputs\hybrid\test_errors.csv`

## 7. 结论

MindSpore 版本已经覆盖特征对比、不平衡训练、概率校准、阈值选择、置信度输出和错例导出，可作为课程交付代码基础。