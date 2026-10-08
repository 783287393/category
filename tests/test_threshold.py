"""自主功能测试（要求 3 个）：阈值调整的量化证据支持设计目标。"""
from __future__ import annotations

import json

import numpy as np

from app import threshold_curve

ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]


def _load_data(key: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((ROOT_DIR / "models" / "threshold_data.json").read_text(encoding="utf-8"))
    return np.asarray(data[f"y_true_{key}"]), np.asarray(data[f"prob_{key}"])


def test_threshold_data_exists_and_models_named() -> None:
    """阈值评估数据存在，且包含两模型、三类概率（供 A/B 对比）。"""
    data = json.loads((ROOT_DIR / "models" / "threshold_data.json").read_text(encoding="utf-8"))
    assert data["models"]["a"] == "LogisticRegression"
    assert data["models"]["b"] == "RandomForest"
    for key in ("a", "b"):
        assert len(data[f"y_true_{key}"]) == len(data[f"prob_{key}"])
        prob = np.asarray(data[f"prob_{key}"])
        assert prob.shape[1] == 3


def test_higher_threshold_precision_does_not_decrease() -> None:
    """提高阈值时 Precision 不下降（量化证据：默认 0.5 vs 高阈值）。"""
    y_true, prob = _load_data("a")
    thresholds = np.round(np.linspace(0.1, 0.9, 17), 2)
    curve = threshold_curve(y_true, prob, target_class=1, thresholds=thresholds)
    p50 = curve.iloc[(curve["threshold"] - 0.50).abs().argmin()]["precision"]
    p90 = curve.iloc[(curve["threshold"] - 0.90).abs().argmin()]["precision"]
    assert p90 >= p50 - 1e-9


def test_higher_threshold_recall_decreases_or_stays() -> None:
    """提高阈值时 Recall 不上升（漏报增加是阈值调高的代价）。"""
    y_true, prob = _load_data("a")
    thresholds = np.round(np.linspace(0.1, 0.9, 17), 2)
    curve = threshold_curve(y_true, prob, target_class=1, thresholds=thresholds)
    r50 = curve.iloc[(curve["threshold"] - 0.50).abs().argmin()]["recall"]
    r90 = curve.iloc[(curve["threshold"] - 0.90).abs().argmin()]["recall"]
    assert r90 <= r50 + 1e-9
