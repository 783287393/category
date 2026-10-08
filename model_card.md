# 模型卡（Model Card）

## 模型概述
- **名称**：Wine 葡萄酒品种三分类模型
- **类型**：完整 scikit-learn Pipeline（StandardScaler → LogisticRegression）
- **任务**：多分类（3 类），预测葡萄酒品种
- **版本/Commit**：见 `models/model_meta.json` 的 `git_commit`

## 用途（Intended Use）
- 对 13 项葡萄酒化学指标进行分类，辅助品种鉴别；
- 作为《开源软件与新技术》实验 04 的教学演示，展示"训练 → 比较 → 保存 → Web 实时推理"的工程化链路。

## 非用途（Out-of-Scope / 不应使用的场景）
- **不可用于实际生产品控、交易或法律依据**：模型基于 178 样本小数据训练，且概率未经校准；
- 不可用于其他品种/产地的葡萄酒（分布外输入，模型未见过）；
- 不可将预测概率解读为事实确定性。

## 样本单位与标签
- 样本单位：一瓶葡萄酒的 13 项化学指标（一次实验室测量）
- 标签：class_0 / class_1 / class_2（UCI 对应 Barolo / Grignolino / Barbera）
- 类别分布：59 / 71 / 48（基本平衡，无需重采样）

## 特征
13 个数值特征（见 `src/features.py`）：alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315_of_diluted_wines, proline

## 数据与划分
- 数据源：UCI Wine Recognition（https://archive.ics.uci.edu/dataset/109/wine），CC BY 4.0
- 划分：分层 80/20（random_state=42），测试集仅用于最终一次评价
- 交叉验证：StratifiedKFold 5 折（shuffle，seed=42），仅用于模型选择

## 主指标与性能
- 主指标：macro-F1（平等看待三个类别）
- 交叉验证对比（训练部分，5 折均值 ± 标准差）：

| 模型 | Accuracy | macro-Precision | macro-Recall | macro-F1 |
|---|---|---|---|---|
| LogisticRegression | 0.979 ± 0.031 | 0.979 | 0.983 | 0.980 ± 0.031 |
| RandomForest | 0.979 ± 0.031 | 0.979 | 0.983 | 0.979 ± 0.032 |

- 最终测试集（仅一次）：Accuracy 0.972，macro-Precision 0.978，macro-Recall 0.967，macro-F1 0.971
- 混淆矩阵（测试集 36 样本）：

```
[[12, 0, 0],
 [ 0, 14, 0],
 [ 0,  1, 9]]
```

## 风险与限制
- 小样本（178），泛化能力有限；
- 概率未经校准，模型可能"看起来自信但实际出错"；
- 分布外输入（OOD）只提示不拒绝，极端新分布下应重训；
- joblib 反序列化与 scikit-learn 版本强相关，跨主版本需在相同环境重训。

## 伦理与合规
- 仅用于课程教学演示；数据按 UCI CC BY 4.0 使用并记录来源；
- 不涉及个人隐私数据，无偏见性敏感维度。
