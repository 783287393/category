"""Wine 葡萄酒品种智能分类 - Streamlit 推理应用。

功能：
  1) 单条预测：表单输入 13 个特征 → 输入校验（缺失/类型/越界）→ 实时预测，
     显示预测类别、概率与模型信息；仅加载固定模型，不在界面重新训练。
  2) 自主功能 - 可调分类阈值：对比两模型的 Precision/Recall 随阈值变化，
     并提供量化证据（基于交叉验证折内结果）。
  3) 自主功能 - 批量 CSV 预测：逐行校验并返回每行错误，结果可下载。
  4) 自主功能 - 输入越界提示：超出训练数据实际范围的输入给出 OOD 警告。

运行：
  .\\.venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    FEATURE_DISPLAY_NAMES,
    FEATURE_NAMES,
    FORM_RANGES,
)
from src.validate import validate_single_row

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

DISCLAIMER = "⚠️ 仅用于课程演示：预测结果不代表事实确定性，分布外输入仅供参考。"


@st.cache_resource
def load_model():
    """只加载固定模型，不在每次点击时重新训练。"""
    meta_path = MODELS_DIR / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = joblib.load(ROOT / meta["model_file"])
    return model, meta


@st.cache_resource
def load_threshold_data():
    path = MODELS_DIR / "threshold_data.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predict_one(model, row: dict) -> tuple[int, np.ndarray]:
    frame = pd.DataFrame([{k: row.get(k, np.nan) for k in FEATURE_NAMES}])
    proba = model.predict_proba(frame)[0]
    pred = int(np.argmax(proba))
    return pred, proba


def threshold_curve(y_true: np.ndarray, prob: np.ndarray, target_class: int, thresholds: np.ndarray):
    """对目标类别计算不同阈值下的 Precision / Recall / F1（正类 = target_class）。"""
    y_bin = (y_true == target_class).astype(int)
    scores = prob[:, target_class]
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_pred == 1) & (y_bin == 1)).sum())
        fp = int(((y_pred == 1) & (y_bin == 0)).sum())
        fn = int(((y_pred == 0) & (y_bin == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def render_single_prediction(model, meta):
    st.subheader("🍷 单条实时预测")
    st.caption("输入葡萄酒的 13 项化学指标（数值），点击预测。表单字段与训练 Schema 完全一致。")

    with st.form("wine_form"):
        cols = st.columns(3)
        inputs: dict[str, float] = {}
        for i, feat in enumerate(FEATURE_NAMES):
            lo, hi = FORM_RANGES[feat]
            val = cols[i % 3].number_input(
                f"{FEATURE_DISPLAY_NAMES[feat]}",
                min_value=None, max_value=None,
                value=round((lo + hi) / 2, 3),
                format="%.3f",
                help=f"训练数据范围约 [{lo:.2f}, {hi:.2f}]",
            )
            inputs[feat] = val
        submitted = st.form_submit_button("预测类别")

    if submitted:
        feature_ranges = meta.get("feature_ranges")
        errors, warnings = validate_single_row(inputs, feature_ranges)
        if errors:
            st.error("输入未通过校验：" + "；".join(errors))
        else:
            for w in warnings:
                st.warning(w)
            pred, proba = predict_one(model, inputs)
            st.success(f"预测类别：**{CLASS_DISPLAY_NAMES[pred]}**")
            prob_df = pd.DataFrame(
                {"类别": CLASS_DISPLAY_NAMES, "概率": proba}
            ).set_index("类别")
            st.bar_chart(prob_df)
            st.markdown(
                f"模型：`{meta['model']}` ｜ 版本：`{meta['git_commit'][:8] or 'local'}`"
                f" ｜ 训练时间：`{meta['trained_at'][:16]}`"
            )
            st.caption(DISCLAIMER)


def render_threshold_view():
    st.subheader("⚖️ 自主功能：可调分类阈值")
    st.markdown(
        "**解决的用户决策问题**：当用户对误报/漏报代价敏感时（例如只接受高置信的 Barolo 推荐），"
        "需要按业务代价调整分类阈值，而不是固定使用默认 0.5。"
    )
    data = load_threshold_data()
    if data is None:
        st.info("未找到阈值评估数据，请先运行 `python -m src.train`。")
        return

    model_choice = st.radio("选择模型曲线", ["a（LogisticRegression）", "b（RandomForest）"], horizontal=True)
    key = "a" if model_choice.startswith("a") else "b"
    target_idx = st.selectbox(
        "目标类别（作为正类）", range(3),
        format_func=lambda i: CLASS_DISPLAY_NAMES[i],
    )
    y_true = np.asarray(data[f"y_true_{key}"])
    prob = np.asarray(data[f"prob_{key}"])
    thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    curve = threshold_curve(y_true, prob, target_idx, thresholds)

    st.line_chart(curve.set_index("threshold"))

    chosen = st.slider("阈值", 0.05, 0.95, 0.50, 0.05)
    row = curve.iloc[(curve["threshold"] - chosen).abs().argmin()]
    st.metric("Precision", f"{row['precision']:.3f}")
    st.metric("Recall", f"{row['recall']:.3f}")
    st.metric("F1", f"{row['f1']:.3f}")
    default_row = curve.iloc[(curve["threshold"] - 0.50).abs().argmin()]
    st.markdown(
        f"**量化证据（交叉验证折内，n={len(y_true)}）**：阈值 0.50 → "
        f"Precision {default_row['precision']:.3f} / Recall {default_row['recall']:.3f}；"
        f"阈值 {chosen:.2f} → Precision {row['precision']:.3f} / Recall {row['recall']:.3f}。"
        "提高阈值会牺牲 Recall（漏报增加）换取 Precision（误报减少），业务上需按误报/漏报代价权衡。"
    )
    st.caption("说明：该曲线来自交叉验证折内概率，不涉及最终测试集；图中指标是训练过程的模型比较证据，不代表部署后效果。")


def render_batch_view(model, meta):
    st.subheader("📦 批量 CSV 预测（自主功能）")
    st.markdown(
        "上传一个 CSV，必须包含与训练一致的 13 列特征（列名见下方提示）。"
        "每行独立校验，坏行只标记错误，不影响其余行；结果可下载。"
    )
    st.code("alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280/od315_of_diluted_wines,proline")

    uploaded = st.file_uploader("上传 CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception:
        st.error("无法解析 CSV，请确认文件编码为 UTF-8。")
        return

    feature_ranges = meta.get("feature_ranges")
    results: list[dict] = []
    for idx, row in frame.iterrows():
        record = {k: row.get(k) for k in FEATURE_NAMES}
        errors, warnings = validate_single_row(record, feature_ranges)
        out: dict = {"__row": idx}
        if errors:
            out["__error"] = "；".join(errors)
            out["__warning"] = ""
            out["predicted_class"] = ""
            out["predicted_class_name"] = ""
            out["class_0_prob"] = ""
            out["class_1_prob"] = ""
            out["class_2_prob"] = ""
        else:
            out["__error"] = ""
            out["__warning"] = "；".join(warnings)
            pred, proba = predict_one(model, {k: record[k] for k in FEATURE_NAMES})
            out["predicted_class"] = int(pred)
            out["predicted_class_name"] = CLASS_NAMES[int(pred)]
            out["class_0_prob"] = float(proba[0])
            out["class_1_prob"] = float(proba[1])
            out["class_2_prob"] = float(proba[2])
        results.append(out)

    result_df = pd.DataFrame(results).set_index("__row")
    result_df.index.name = "row_index"
    n_ok = int((result_df["__error"] == "").sum())
    n_bad = int(len(result_df) - n_ok)
    st.success(f"共 {len(result_df)} 行：预测成功 {n_ok} 行，校验失败 {n_bad} 行（失败行已标记，不影响其余行）。")

    st.dataframe(result_df)
    buf = io.StringIO()
    result_df.to_csv(buf, index=True)
    st.download_button(
        "下载预测结果 CSV",
        data=buf.getvalue().encode("utf-8-sig"),
        file_name="wine_predictions.csv",
        mime="text/csv",
    )


def render_model_info(meta):
    st.subheader("ℹ️ 模型信息")
    m = st.container(border=True)
    m.markdown(
        f"- **模型**：`{meta['model']}`（完整 Pipeline：{', '.join(meta['pipeline'])}）\n"
        f"- **任务**：{meta['task']}\n"
        f"- **主指标**：{meta['main_metric']}（宏平均 F1，平等看待三个类别）\n"
        f"- **最终测试集（仅评价一次）**：Accuracy {meta['final_metrics']['accuracy']:.3f}，"
        f"macro-Precision {meta['final_metrics']['precision_macro']:.3f}，"
        f"macro-Recall {meta['final_metrics']['recall_macro']:.3f}，"
        f"macro-F1 {meta['final_metrics']['f1_macro']:.3f}\n"
        f"- **数据**：{meta['dataset']['name']}（{meta['dataset']['n_samples']} 样本 / "
        f"{meta['dataset']['n_features']} 特征 / {meta['dataset']['n_classes']} 类）\n"
        f"- **划分**：分层 80/20（random_state={meta['split']['random_state']}）\n"
        f"- **交叉验证**：{meta['cv']['method']}（{meta['cv']['n_splits']} 折）\n"
        f"- **scikit-learn**：{meta['sklearn_version']} ｜ **Python**：{meta['python_version']}\n"
        f"- **训练时间**：{meta['trained_at'][:16]} ｜ **Git Commit**：`{meta['git_commit'][:12]}`\n"
        f"- **模型 SHA256**：`{meta['model_sha256'][:16]}…`"
    )
    st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="Wine 智能分类系统", page_icon="🍷", layout="wide")
    st.title("🍷 Wine 葡萄酒品种智能分类系统")
    st.markdown(
        "《开源软件与新技术》实验 04 —— 机器学习智能分类系统。"
        "基于 **scikit-learn 1.7.2** 训练与 **Streamlit 1.50.0** 封装，模型固定加载，实时推理。"
    )

    model, meta = load_model()
    tab1, tab2, tab3, tab4 = st.tabs(["单条预测", "阈值分析（自主功能）", "批量预测（自主功能）", "模型信息"])

    with tab1:
        render_single_prediction(model, meta)
    with tab2:
        render_threshold_view()
    with tab3:
        render_batch_view(model, meta)
    with tab4:
        render_model_info(meta)


if __name__ == "__main__":
    main()
