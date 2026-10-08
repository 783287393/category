"""pytest 共享 fixture：合法样本、已保存模型与元数据。"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from src.features import FEATURE_NAMES

ROOT_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def wine_frame() -> pd.DataFrame:
    wine = load_wine(as_frame=True)
    return wine.data


@pytest.fixture(scope="session")
def valid_row() -> dict:
    """取 Wine 数据第一行作为合法样本。"""
    wine = load_wine(as_frame=True)
    return {col: float(wine.data.iloc[0][col]) for col in FEATURE_NAMES}


@pytest.fixture(scope="session")
def model():
    return joblib.load(ROOT_DIR / "models" / "model.joblib")


@pytest.fixture(scope="session")
def meta() -> dict:
    return json.loads((ROOT_DIR / "models" / "model_meta.json").read_text(encoding="utf-8"))
