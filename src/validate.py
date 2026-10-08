"""输入校验：在预测前检查缺失、类型、未知字段与越界值。

规则：
- 缺失/非数值 → 拒绝（返回错误信息）
- 未知/缺失列 → 拒绝
- 超出训练数据实际范围 → 提示分布外（OOD）警告，仍允许预测但展示限制
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.features import FEATURE_NAMES


def validate_input_frame(frame: pd.DataFrame, feature_ranges: dict[str, dict[str, float]] | None = None) -> tuple[list[str], list[str]]:
    """校验一个 DataFrame（每行一条样本）。

    返回 (errors, warnings)：
    - errors: 会导致该行无法预测的问题列表
    - warnings: 不阻断预测的提示（如越界值）
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in FEATURE_NAMES if c not in frame.columns]
    if missing_cols:
        errors.append(f"缺少列: {missing_cols}")
    extra_cols = [c for c in frame.columns if c not in FEATURE_NAMES]
    if extra_cols:
        errors.append(f"存在未知列: {extra_cols}")

    if errors:
        return errors, warnings

    for col in FEATURE_NAMES:
        series = frame[col]
        n_missing = int(series.isna().sum())
        if n_missing > 0:
            errors.append(f"列 {col} 存在 {n_missing} 个缺失值（不允许空值）")
        try:
            numeric = pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError):
            errors.append(f"列 {col} 包含无法转换为数值的内容")
            continue
        if feature_ranges and col in feature_ranges:
            lo = feature_ranges[col].get("min")
            hi = feature_ranges[col].get("max")
            if lo is not None and hi is not None:
                out_of_range = ((numeric < lo) | (numeric > hi)).sum()
                if out_of_range > 0:
                    warnings.append(
                        f"列 {col} 有 {int(out_of_range)} 个值超出训练数据范围 [{lo:.3f}, {hi:.3f}]，"
                        "属于分布外输入，预测仅供参考"
                    )
    return errors, warnings


def validate_single_row(row: dict[str, Any], feature_ranges: dict[str, dict[str, float]] | None = None) -> tuple[list[str], list[str]]:
    """校验单条输入（表单场景）。返回 (errors, warnings)。"""
    frame = pd.DataFrame([{k: v for k, v in row.items()}])
    return validate_input_frame(frame, feature_ranges)
