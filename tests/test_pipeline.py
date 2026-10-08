"""Pipeline 测试（要求 6 个）：缺失、未知类别、保存/加载、列顺序。"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES
from src.validate import validate_input_frame, validate_single_row

ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_raw_schema_dataframe_predicts(model, wine_frame: pd.DataFrame) -> None:
    """原始 Schema 的 DataFrame 可直接 predict，无需手工预处理。"""
    y_pred = model.predict(wine_frame.head(10))
    assert y_pred.shape == (10,)
    assert set(np.unique(y_pred)).issubset({0, 1, 2})


def test_save_load_prediction_consistency(model, wine_frame: pd.DataFrame, meta: dict) -> None:
    """模型保存前后对同一 fixture 的预测一致（保存/加载不改变行为）。"""
    reloaded = joblib.load(ROOT_DIR / meta["model_file"])
    a = model.predict(wine_frame.head(20))
    b = reloaded.predict(wine_frame.head(20))
    assert (a == b).all()


def test_unknown_column_rejected(wine_frame: pd.DataFrame, meta: dict) -> None:
    """未知列被拒绝。"""
    bad = wine_frame.head(1).copy()
    bad["hacked_column"] = 1.0
    errors, _ = validate_input_frame(bad, meta.get("feature_ranges"))
    assert any("未知列" in e for e in errors)


def test_missing_value_rejected(wine_frame: pd.DataFrame, meta: dict) -> None:
    """缺失值被拒绝。"""
    bad = wine_frame.head(1).copy()
    bad.loc[bad.index[0], "alcohol"] = np.nan
    errors, _ = validate_input_frame(bad, meta.get("feature_ranges"))
    assert any("缺失" in e for e in errors)


def test_non_numeric_rejected(wine_frame: pd.DataFrame, meta: dict) -> None:
    """非数值内容被拒绝。"""
    bad = wine_frame.head(1).copy()
    bad["proline"] = bad["proline"].astype(object)
    bad.loc[bad.index[0], "proline"] = "not-a-number"
    errors, _ = validate_input_frame(bad, meta.get("feature_ranges"))
    assert any("数值" in e for e in errors)


def test_feature_order_independent(model, wine_frame: pd.DataFrame) -> None:
    """列顺序变化不影响预测（按列名定位，符合训练 Schema）。"""
    ordered = wine_frame.head(5)[FEATURE_NAMES]
    shuffled = wine_frame.head(5)[FEATURE_NAMES[::-1]]
    assert (model.predict(ordered) == model.predict(shuffled)).all()


def test_model_file_and_meta_sha256(meta: dict) -> None:
    """模型文件存在、元数据完整，且 SHA256 与记录一致。"""
    import hashlib

    path = ROOT_DIR / meta["model_file"]
    assert path.exists()
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == meta["model_sha256"]
    assert "git_commit" in meta and "trained_at" in meta and "features" in meta
