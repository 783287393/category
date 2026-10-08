"""训练脚本：Wine 三分类任务，比较两种算法并保存最佳 Pipeline。

流程（与 README 架构图一致）：
  原始数据集 → Schema/质量检查 → 训练/验证/测试划分
    → Pipeline A(StandardScaler+LogisticRegression)
    → Pipeline B(StandardScaler+RandomForest)
    → 5 折分层交叉验证（仅在训练部分）→ 按主指标选择模型
    → 在完整训练集上拟合一次 → 测试集仅评价一次
    → 保存 model.joblib + model_meta.json + 阈值评估数据

运行：
  .\\.venv\\Scripts\\python.exe -m src.train
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import CLASS_NAMES, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAIN_METRIC = "f1_macro"  # 主指标：各类别 F1 的宏平均，平等看待三个类别

DATASET_INFO = {
    "name": "Wine（UCI Wine Recognition）",
    "source": "https://archive.ics.uci.edu/dataset/109/wine",
    "license": "CC BY 4.0（按 UCI 页面记录）",
    "n_samples": 178,
    "n_features": 13,
    "n_classes": 3,
}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "not-committed"
    except Exception:
        return "not-committed"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pipeline(estimator) -> Pipeline:
    """预处理器与模型保存在同一 Pipeline，训练和推理不会各写一套规则。"""
    numeric_pipe = Pipeline([("scale", StandardScaler())])
    preprocess = ColumnTransformer([("num", numeric_pipe, FEATURE_NAMES)])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def cv_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray):
    """5 折分层交叉验证，记录每折指标与验证折概率（供阈值曲线使用）。

    只使用训练部分的折内数据，测试集不参与。
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    prob_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    y_arr = np.asarray(y)  # 统一按位置索引取值（折索引为位置索引）
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        pipe = clone(pipeline)
        pipe.fit(X.iloc[tr], y_arr[tr])
        y_pred = pipe.predict(X.iloc[va])
        rows.append({
            "fold": fold,
            "accuracy": float(accuracy_score(y_arr[va], y_pred)),
            "precision_macro": float(precision_score(y_arr[va], y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_arr[va], y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_arr[va], y_pred, average="macro", zero_division=0)),
        })
        prob_parts.append(pipe.predict_proba(X.iloc[va]))
        y_parts.append(y_arr[va])
    summary = pd.DataFrame(rows)
    return summary, np.vstack(prob_parts), np.concatenate(y_parts)


def final_evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """测试集只评价一次：Accuracy / macro、weighted Precision、Recall、F1 + 混淆矩阵。"""
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test_samples": int(len(y_test)),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    # 1. 数据加载与 Schema/质量检查
    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target
    assert X.shape == (178, 13) and X.columns.tolist() == FEATURE_NAMES
    assert set(y.unique()) == {0, 1, 2}

    feature_ranges = {
        col: {"min": float(X[col].min()), "max": float(X[col].max())}
        for col in FEATURE_NAMES
    }

    # 2. 先划分再拟合：分层划分，随机种子固定，测试集最后才看
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # 3. 两条 Pipeline：Logistic Regression 与 Random Forest
    model_a = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    model_b = build_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    )
    names = {"a": "LogisticRegression", "b": "RandomForest"}

    # 4. 交叉验证公平比较（同划分、同指标）
    cv_a, prob_a, y_cv_a = cv_evaluate(model_a, X_train, y_train)
    cv_b, prob_b, y_cv_b = cv_evaluate(model_b, X_train, y_train)

    comparison = pd.DataFrame({
        "model": ["LogisticRegression", "RandomForest"],
        "accuracy_mean": [cv_a["accuracy"].mean(), cv_b["accuracy"].mean()],
        "accuracy_std": [cv_a["accuracy"].std(), cv_b["accuracy"].std()],
        "precision_macro_mean": [cv_a["precision_macro"].mean(), cv_b["precision_macro"].mean()],
        "recall_macro_mean": [cv_a["recall_macro"].mean(), cv_b["recall_macro"].mean()],
        "f1_macro_mean": [cv_a["f1_macro"].mean(), cv_b["f1_macro"].mean()],
        "f1_macro_std": [cv_a["f1_macro"].std(), cv_b["f1_macro"].std()],
    })
    print("\n===== 交叉验证比较（5 折分层，训练部分）=====")
    print(comparison.to_string(index=False))

    # 5. 按主指标选择模型（不接触测试集）
    mean_f1 = comparison["f1_macro_mean"]
    best_name = names["a"] if mean_f1[0] >= mean_f1[1] else names["b"]
    best = clone(model_a) if best_name == "LogisticRegression" else clone(model_b)
    best.fit(X_train, y_train)

    # 6. 测试集一次最终评价
    final = final_evaluate(best, X_test, y_test)
    print(f"\n===== 最终测试集评价（仅此一次）: {best_name} =====")
    print(json.dumps(final, ensure_ascii=False, indent=2))

    # 7. 保存完整 Pipeline 与元数据
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(best, model_path)
    model_sha = sha256_of(model_path)

    meta = {
        "model": best_name,
        "task": "Wine 葡萄酒品种三分类",
        "main_metric": MAIN_METRIC,
        "dataset": DATASET_INFO,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "split": {"strategy": "stratified train_test_split", "test_size": TEST_SIZE, "random_state": SEED},
        "cv": {"method": "StratifiedKFold", "n_splits": CV_FOLDS, "shuffle": True, "random_state": SEED},
        "features": FEATURE_NAMES,
        "feature_ranges": feature_ranges,
        "classes": CLASS_NAMES,
        "final_metrics": final,
        "comparison": comparison.round(4).to_dict(orient="records"),
        "pipeline": [s for s in best.named_steps.keys()],
        "sklearn_version": joblib.__version__ and str(__import__("sklearn").__version__),
        "python_version": __import__("sys").version.split()[0],
        "git_commit": git_commit(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_file": "models/model.joblib",
        "model_sha256": model_sha,
        "limits": "仅用于课程演示；概率不等于事实确定性；分布外输入仅供参考。",
    }
    (MODELS_DIR / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 8. 阈值评估数据（来自交叉验证折，供界面"可调阈值"自主功能使用）
    threshold_data = {
        "classes": CLASS_NAMES,
        "y_true_a": y_cv_a.tolist(), "prob_a": prob_a.tolist(),
        "y_true_b": y_cv_b.tolist(), "prob_b": prob_b.tolist(),
        "models": {"a": "LogisticRegression", "b": "RandomForest"},
    }
    (MODELS_DIR / "threshold_data.json").write_text(
        json.dumps(threshold_data, ensure_ascii=False), encoding="utf-8"
    )

    # 9. 保留评价证据（README / PPT 素材）
    (DOCS_DIR / "model_comparison.json").write_text(
        json.dumps(comparison.round(4).to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS_DIR / "final_evaluation.json").write_text(
        json.dumps({"model": best_name, **final}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n已保存: {model_path}")
    print(f"已保存: {MODELS_DIR / 'model_meta.json'}")
    print(f"已保存: {MODELS_DIR / 'threshold_data.json'}")


if __name__ == "__main__":
    main()
