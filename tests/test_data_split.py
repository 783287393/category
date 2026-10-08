"""数据/划分测试（要求 5 个）：Schema、标签、分层/分组、泄漏。"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES

SEED = 42


def test_wine_shape_and_schema(wine_frame: pd.DataFrame) -> None:
    """Wine 数据为 178 样本、13 特征、3 类，列名与训练 Schema 一致。"""
    wine = load_wine(as_frame=True)
    assert wine_frame.shape == (178, 13)
    assert wine_frame.columns.tolist() == FEATURE_NAMES
    assert set(wine.target.unique()) == {0, 1, 2}


def test_no_missing_values(wine_frame: pd.DataFrame) -> None:
    """训练数据无缺失值。"""
    assert wine_frame.isna().sum().sum() == 0


def test_split_reproducible(wine_frame: pd.DataFrame) -> None:
    """固定随机种子下划分结果可复现。"""
    wine = load_wine(as_frame=True)
    X1, _, y1, _ = train_test_split(wine.data, wine.target, test_size=0.2, stratify=wine.target, random_state=SEED)
    X2, _, y2, _ = train_test_split(wine.data, wine.target, test_size=0.2, stratify=wine.target, random_state=SEED)
    assert X1.index.equals(X2.index)
    assert (y1.values == y2.values).all()


def test_stratified_class_distribution(wine_frame: pd.DataFrame) -> None:
    """分层划分保持类别比例（与全量分布基本一致）。"""
    wine = load_wine(as_frame=True)
    _, X_test, _, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, stratify=wine.target, random_state=SEED
    )
    full_dist = wine.target.value_counts(normalize=True)
    test_dist = pd.Series(y_test).value_counts(normalize=True)
    for cls in (0, 1, 2):
        assert abs(full_dist[cls] - test_dist[cls]) < 0.06


def test_no_train_test_overlap(wine_frame: pd.DataFrame) -> None:
    """训练/测试样本不交叉（无泄漏）；Wine 无重复主体，按样本划分即可。"""
    wine = load_wine(as_frame=True)
    X_train, X_test, _, _ = train_test_split(
        wine.data, wine.target, test_size=0.2, stratify=wine.target, random_state=SEED
    )
    overlap = set(X_train.index).intersection(set(X_test.index))
    assert len(overlap) == 0
    assert len(X_train.index) + len(X_test.index) == 178
