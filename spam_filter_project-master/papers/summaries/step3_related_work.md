# Step 3 文献整理

本项目固定阅读并总结以下九篇文献：

1. Sahami et al. 1998
2. Drucker et al. 1999
3. Androutsopoulos et al. 2000
4. Metsis et al. 2006
5. Guzella & Caminhas 2009
6. Platt 1999
7. Niculescu-Mizil & Caruana 2005
8. Guo et al. 2017
9. Devlin et al. 2019

## 主要启发

- 经典文本过滤以 Naive Bayes、SVM、LR 为主线。
- 概率输出不能只看分类分数，必须做校准与可靠性分析。
- 现代深度模型需要温度校准，否则概率常常过度自信。
- 垃圾邮件检测必须在不平衡条件下看 PR-AUC、Recall、F1、ECE。

## 对本项目的直接影响

- 采用词级 TF-IDF 作为 NB / LR / SVM 的主输入。
- 采用字符级 TF-IDF 处理拼写变形与欺骗字符。
- 采用结构化特征增强可解释性。
- 对 SVM、树模型和 BERT 统一做校准。
