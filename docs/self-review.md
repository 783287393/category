# 自我 Code Review 记录（合并前检查清单）

Branch: `feature/model-application` → 合并到 `main`
Date: 2026-10-08

## 检查清单

| # | 检查项 | 结果 | 说明 |
|---|---|---|---|
| 1 | 测试集是否只用于最终一次评价 | ✅ | 交叉验证仅在训练部分（X_train 内 5 折），测试集仅在 `src/train.py` 的 `final_evaluate` 使用一次 |
| 2 | 预处理器是否只在训练折拟合 | ✅ | 全部通过 Pipeline 拟合；`validate.py`/`app.py` 不做任何手工预处理 |
| 3 | 主指标与平均方式是否说明 | ✅ | macro-F1，README 与 model_card 均说明"平等看待三个类别" |
| 4 | 特征数量错误防护 | ✅ | 表单字段来自 `FEATURE_NAMES`；`validate_input_frame` 校验未知/缺失列；测试 `test_feature_order_independent` 覆盖列顺序 |
| 5 | 训练时未见类别 | ✅ | Wine 全为数值特征，无类别编码问题；OOD 输入给出警告 |
| 6 | 模型加载安全 | ✅ | 只加载本人训练的 `models/model.joblib`（SHA256 记录在元数据），界面不接收用户模型文件 |
| 7 | 阈值/解释限制声明 | ✅ | 阈值曲线基于交叉验证折内概率，界面注明不涉及最终测试集；概率展示"仅用于课程演示" |
| 8 | 批量坏行隔离 | ✅ | `app.py` 批量页逐行校验，坏行标记错误不影响其余行；`data/sample_input_bad_row.csv` 演示 |
| 9 | 依赖固定 | ✅ | `requirements.txt` 由 `pip freeze` 生成；README 记录实测版本 |
| 10 | 许可证合规 | ✅ | NOTICE.md 记录 scikit-learn（BSD-3）、Streamlit（Apache-2.0）、Wine 数据（CC BY 4.0）；本项目 MIT |

## 审查发现与处理
- **发现 1**：初版测试中 `meta["model_file"]` 与 `ROOT/models/` 重复拼接导致路径错误 → 已修复为 `ROOT / meta["model_file"]`。
- **发现 2**：pandas 对 float64 列赋字符串触发 FutureWarning → 已先转 object 类型，消除警告。
- **发现 3**：交叉验证折索引为位置索引而 `y` 为保留原索引的 Series → 改为 `np.asarray(y)` 按位置取值。

## 结论
- 检查项全部通过，可合并。
- 遗留风险（不阻塞合并）：模型概率未校准；OOD 仅提示不拒绝；依赖跨主版本需重训（已写入 README 已知问题）。
