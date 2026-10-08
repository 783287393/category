"""Wine 数据集特征 Schema 定义。

供训练脚本、推理界面与测试共用，保证"训练 Schema == 推理 Schema"。
"""
from __future__ import annotations

# sklearn.datasets.load_wine 返回的 13 个数值特征（顺序即训练顺序）
FEATURE_NAMES: list[str] = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

# 特征中文说明（用于界面提示；来源：UCI Wine 数据集变量说明）
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "alcohol": "酒精含量",
    "malic_acid": "苹果酸",
    "ash": "灰分",
    "alcalinity_of_ash": "灰分的碱度",
    "magnesium": "镁含量",
    "total_phenols": "总酚",
    "flavanoids": "黄酮类化合物",
    "nonflavanoid_phenols": "非黄酮类酚",
    "proanthocyanins": "原花青素",
    "color_intensity": "颜色强度",
    "hue": "色调",
    "od280/od315_of_diluted_wines": "稀释葡萄酒 OD280/OD315 比值",
    "proline": "脯氨酸",
}

# 类别展示名（sklearn 内部标签为 0/1/2；UCI 数据集对应葡萄品种）
CLASS_NAMES: list[str] = ["class_0", "class_1", "class_2"]
CLASS_DISPLAY_NAMES: list[str] = ["Barolo（类0）", "Grignolino（类1）", "Barbera（类2）"]

# 界面数值输入框的合理范围（用于表单提示；严格的越界判定以训练数据
# 实际 min/max 为准，保存在 model_meta.json["feature_ranges"] 中）
FORM_RANGES: dict[str, tuple[float, float]] = {
    "alcohol": (10.0, 15.0),
    "malic_acid": (0.0, 6.0),
    "ash": (1.0, 4.0),
    "alcalinity_of_ash": (8.0, 30.0),
    "magnesium": (60.0, 170.0),
    "total_phenols": (0.5, 4.0),
    "flavanoids": (0.0, 5.5),
    "nonflavanoid_phenols": (0.0, 1.0),
    "proanthocyanins": (0.0, 4.0),
    "color_intensity": (0.5, 13.0),
    "hue": (0.3, 1.8),
    "od280/od315_of_diluted_wines": (1.0, 4.0),
    "proline": (250.0, 1700.0),
}
