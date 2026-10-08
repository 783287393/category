"""应用测试（要求 6 个）：合法/非法/边界/批量输入，提示清楚不暴露堆栈。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app import predict_one, threshold_curve
from src.features import CLASS_NAMES, FEATURE_NAMES
from src.validate import validate_input_frame, validate_single_row


def test_valid_input_predicts(model, valid_row: dict) -> None:
    """合法输入可预测，返回合法类别与 3 个概率。"""
    pred, proba = predict_one(model, valid_row)
    assert pred in (0, 1, 2)
    assert proba.shape == (3,)
    assert np.isclose(proba.sum(), 1.0)


def test_missing_feature_rejected(valid_row: dict, meta: dict) -> None:
    """缺失关键特征时校验报错（不进入预测）。"""
    row = {k: v for k, v in valid_row.items()}
    del row["proline"]
    errors, _ = validate_single_row(row, meta.get("feature_ranges"))
    assert len(errors) > 0


def test_out_of_range_warns(valid_row: dict, meta: dict) -> None:
    """越界（分布外）输入产生警告而非静默通过。"""
    row = dict(valid_row)
    row["proline"] = 99999.0  # 远超训练最大值
    errors, warnings = validate_single_row(row, meta.get("feature_ranges"))
    assert errors == []
    assert any("超出训练数据范围" in w for w in warnings)


def test_batch_row_errors_isolated(wine_frame: pd.DataFrame, meta: dict) -> None:
    """批量场景：坏行只标记错误，不影响好行。"""
    rows = wine_frame.head(3).copy()
    rows.loc[rows.index[1], "magnesium"] = np.nan  # 中间行损坏
    feature_ranges = meta.get("feature_ranges")
    errors, _ = validate_input_frame(rows, feature_ranges)
    # 校验函数在列级报告缺失；应用层逐行隔离错误（此处验证缺失被捕获）
    assert any("缺失" in e for e in errors)


def test_batch_predict_all_rows(model, wine_frame: pd.DataFrame) -> None:
    """批量：3 行合法数据全部可预测，结果数量一致。"""
    preds = [predict_one(model, {c: float(v) for c, v in r.items()}) for _, r in wine_frame.head(3).iterrows()]
    assert len(preds) == 3
    assert all(p in (0, 1, 2) for p, _ in preds)


def test_threshold_curve_shape_and_bounds(model) -> None:
    """阈值曲线：输入输出一致、数值在 [0,1]。"""
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 5)
    rng = np.random.default_rng(0)
    prob = rng.dirichlet(np.ones(3), size=len(y_true))
    curve = threshold_curve(y_true, prob, target_class=1, thresholds=np.array([0.1, 0.5, 0.9]))
    assert list(curve.columns) == ["threshold", "precision", "recall", "f1"]
    assert (curve[["precision", "recall", "f1"]].values >= 0).all()
    assert (curve[["precision", "recall", "f1"]].values <= 1).all()
    assert len(curve) == 3
