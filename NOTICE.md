# NOTICE — 第三方来源与许可证清单

本仓库（Wine 葡萄酒品种智能分类系统）是在以下开源项目/资源的基础上二次开发的教学实验作品。学生本人实现训练脚本、输入校验、推理界面、自主功能与测试；依赖库与数据版权归各自作者所有。

| 资源 | 来源 | 许可证 | 用途 |
|---|---|---|---|
| scikit-learn（实测 1.7.2） | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | 训练、Pipeline、评价、模型保存 |
| Streamlit（实测 1.50.0） | https://github.com/streamlit/streamlit | Apache-2.0 | 交互推理界面 |
| Wine 数据集 | https://archive.ics.uci.edu/dataset/109/wine | CC BY 4.0（按 UCI 页面记录） | 训练与演示数据 |
| 依赖包（pandas/NumPy/joblib/pytest 等） | 见 requirements.txt | 见各包许可证 | 数据处理与测试 |

## 说明
- 开源库许可证不替代数据许可证；Wine 数据按 CC BY 4.0 使用，引用时注明来源。
- 本项目未 Fork 任何上游仓库，未复制上游 Logo/主题；作为独立应用仓库，依赖固定版本。
- 本项目对外发布采用 MIT 许可证（与上游 BSD-3 与 Apache-2.0 均兼容）。
- 若分发包含模型文件（models/model.joblib），其训练产物属本人代码运行结果，受 MIT 许可证约束；数据权利仍属 UCI 数据集条款。
