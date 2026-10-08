"""模型评价测试（要求 2 个）：两种算法同协议比较，最终测试集仅评价一次。"""
from __future__ import annotations

import json

from src.features import CLASS_NAMES, FEATURE_NAMES

ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_comparison_same_protocol(meta: dict) -> None:
    """两种算法在同一划分、同指标下比较；指标均在 [0,1]，且都记录均值与方差。"""
    comparison = meta["comparison"]
    assert {c["model"] for c in comparison} == {"LogisticRegression", "RandomForest"}
    for c in comparison:
        for key in ("accuracy_mean", "precision_macro_mean", "recall_macro_mean", "f1_macro_mean"):
            assert 0.0 <= c[key] <= 1.0
        assert c["accuracy_std"] >= 0.0
        assert c["f1_macro_std"] >= 0.0


def test_final_evaluation_recorded_once(meta: dict) -> None:
    """最终测试集指标、混淆矩阵与特征/类别映射完整且形状正确。"""
    final = meta["final_metrics"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
        assert 0.0 <= final[key] <= 1.0
    cm = final["confusion_matrix"]
    assert len(cm) == 3 and all(len(row) == 3 for row in cm)
    assert sum(sum(row) for row in cm) == final["n_test_samples"] == meta["n_test_samples"]
    assert meta["features"] == FEATURE_NAMES
    assert meta["classes"] == CLASS_NAMES
    # 测试集评价记录作为证据文件存在
    evidence = json.loads(
        (ROOT_DIR / "docs" / "final_evaluation.json").read_text(encoding="utf-8")
    )
    assert evidence["model"] == meta["model"]
    assert evidence["accuracy"] == final["accuracy"]
