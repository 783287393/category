# 数据目录说明

本目录存放演示输入数据，**训练数据不在此下载**。

- 训练数据：`sklearn.datasets.load_wine`（内置，178 样本、13 数值特征、3 类，无网络依赖）
- 原始来源：https://archive.ics.uci.edu/dataset/109/wine（UCI Wine Recognition）
- 许可证：CC BY 4.0（按 UCI 页面记录；开源库许可证不替代数据许可证）

## 演示文件
- `sample_input.csv`：5 行合法样本（取自 Wine 数据前 5 行），用于「批量预测」页签演示。
- `sample_input_bad_row.csv`：含 1 行坏数据（缺失值）+ 4 行合法数据，用于演示"坏行隔离、不影响其余行"。
