import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ============================================================
# sklearn 版本兼容补丁
# ============================================================
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


# ============================================================
# 基础路径
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "TabPFN_GPU_ST_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "TabPFN_GPU_ST_preprocessor.joblib"
EXPLAINER_PATH = BASE_DIR / "shap_explainer_TabPFN_GPU_ST.joblib"


# ============================================================
# 原始输入特征（14个）
# ============================================================
RAW_FEATURES = [
    "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
    "Ag2.36", "Ag4.75", "Ag9.5", "FT", "FC", "FL", "TS"
]

DEFAULT_INPUT = {
    "Pe": 70.0,
    "Du": 100.0,
    "SP": 50.0,
    "AC": 5.0,
    "AV": 4.0,
    "VMA": 15.0,
    "VFA": 70.0,
    "Ag2.36": 35.0,
    "Ag4.75": 48.0,
    "Ag9.5": 73.0,
    "FT": "no_fiber",
    "FC": 0.0,
    "FL": 0.0,
    "TS": 0.0,
}

RANGES = {
    "Pe": (0.0, 200.0, 0.1),
    "Du": (0.0, 300.0, 1.0),
    "SP": (0.0, 100.0, 0.1),
    "AC": (0.0, 20.0, 0.1),
    "AV": (0.0, 30.0, 0.01),
    "VMA": (0.0, 80.0, 0.01),
    "VFA": (0.0, 100.0, 0.01),
    "Ag2.36": (0.0, 100.0, 0.01),
    "Ag4.75": (0.0, 100.0, 0.01),
    "Ag9.5": (0.0, 100.0, 0.01),
    "FC": (0.0, 10.0, 0.01),
    "FL": (0.0, 100.0, 0.01),
    "TS": (0.0, 10000.0, 1.0),
}

FALLBACK_FT_OPTIONS = [
    "no_fiber",
    "basalt fiber",
    "glass fiber",
    "polyester fiber",
    "steel fiber",
]

# ============================================================
# 页面设置
# ============================================================
st.set_page_config(page_title="FRAC ST Prediction GUI", layout="wide")
st.markdown(
    """
    <style>
    /* 数值输入框里的数字 */
    [data-testid="stNumberInput"] input {
        font-size: 24px !important;
        font-weight: normal !important;
    }

    /* 下拉框当前选中的文字 */
    div[data-baseweb="select"] > div {
        font-size: 20px !important;
        font-weight: normal !important;
    }

    /* 各输入组件上方的标签文字，如 Pe、Du、FT */
    label[data-testid="stWidgetLabel"] p {
        font-size: 26px !important;
        font-weight: normal !important;
    }

    /* number_input 右侧 ± 按钮 */
    [data-testid="stNumberInput"] button {
        font-size: 22px !important;
        font-weight: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='background-color:#0B5ED7;padding:8px;border-radius:10px;text-align:center;'>
        <h2 style='color:white;margin:0;'>Asphalt concrete Splitting Strength (ST) Prediction and SHAP Analysis</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This interface is built for **14 input features → 1 output feature (ST)** prediction.
    """
)


# ============================================================
# 资源加载
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    info = {
        "model": None,
        "preprocessor": None,
        "explainer": None,
        "errors": {},
    }

    try:
        info["model"] = joblib.load(MODEL_PATH)
    except Exception:
        info["errors"]["model"] = traceback.format_exc()

    try:
        info["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    except Exception:
        info["errors"]["preprocessor"] = traceback.format_exc()

    try:
        info["explainer"] = joblib.load(EXPLAINER_PATH)
    except Exception:
        info["errors"]["explainer"] = traceback.format_exc()

    return info


artifacts = load_artifacts()


# ============================================================
# 工具函数
# ============================================================
def get_feature_names_from_preprocessor(preprocessor):
    if preprocessor is None:
        return None
    try:
        names = preprocessor.get_feature_names_out()
        names = [str(x) for x in names]
        cleaned = []
        for n in names:
            n2 = n.replace("num__", "").replace("cat__", "")
            cleaned.append(n2)
        return cleaned
    except Exception:
        return None


def get_ft_options_from_preprocessor(preprocessor):
    if preprocessor is None:
        return FALLBACK_FT_OPTIONS

    try:
        if hasattr(preprocessor, "transformers_"):
            for name, transformer, cols in preprocessor.transformers_:
                if transformer == "drop":
                    continue
                if isinstance(cols, (list, tuple)) and "FT" in list(cols):
                    if hasattr(transformer, "categories_") and len(transformer.categories_) > 0:
                        return [str(x) for x in transformer.categories_[0].tolist()]
                    if hasattr(transformer, "named_steps"):
                        for _, step in transformer.named_steps.items():
                            if hasattr(step, "categories_") and len(step.categories_) > 0:
                                return [str(x) for x in step.categories_[0].tolist()]
    except Exception:
        pass

    return FALLBACK_FT_OPTIONS


def transform_input(preprocessor, raw_df):
    if preprocessor is None:
        raise RuntimeError("未能加载预处理器，无法进行模型输入转换。")

    X = preprocessor.transform(raw_df)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(X, "toarray"):
        X = X.toarray()

    if feature_names is not None and len(feature_names) == X.shape[1]:
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    return X_df


def safe_predict(model, X_df):
    if model is None:
        raise RuntimeError("未能加载模型，无法进行预测。")
    pred = model.predict(X_df)
    if isinstance(pred, (list, tuple, np.ndarray)):
        return float(np.asarray(pred).reshape(-1)[0])
    return float(pred)


def make_local_shap_explanation(explainer, X_df):
    if explainer is None:
        raise RuntimeError("未能加载 SHAP explainer，无法生成单样本 SHAP 解释。")
    return explainer(X_df)


def plot_waterfall_from_explanation(sample_exp, max_display=12):
    plt.close("all")
    plt.figure(figsize=(4.6, 3.2))
    shap.plots.waterfall(sample_exp, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_force_from_explanation(sample_exp, top_n=8):
    """
    自定义 force-style 图：
    1. 使用真实特征名，而不是 Feature 1/2/3 这类占位符；
    2. 控制画布比例，避免 shap.force_plot(matplotlib=True) 在 Streamlit 中变形；
    3. 仅展示绝对贡献最大的 top_n 个特征，提升可读性。
    """
    plt.close("all")

    values = np.asarray(sample_exp.values).reshape(-1)
    feature_names = list(sample_exp.feature_names) if sample_exp.feature_names is not None else [f"f{i}" for i in range(len(values))]
    feature_data = np.asarray(sample_exp.data).reshape(-1) if getattr(sample_exp, "data", None) is not None else np.array([np.nan] * len(values))
    base_value = float(np.asarray(sample_exp.base_values).reshape(-1)[0])
    pred_value = base_value + float(values.sum())

    top_idx = np.argsort(np.abs(values))[-min(top_n, len(values)):]
    top_idx = top_idx[np.argsort(np.abs(values[top_idx]))[::-1]]

    top_vals = values[top_idx]
    top_names = [feature_names[i] for i in top_idx]
    top_data = feature_data[top_idx]

    labels = []
    for n, d in zip(top_names, top_data):
        if isinstance(d, (float, int, np.floating, np.integer)):
            labels.append(f"{n} = {d:.3g}")
        else:
            labels.append(f"{n} = {d}")

    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    left_neg = base_value
    left_pos = base_value
    y_neg = 0.25
    y_pos = -0.25
    bar_h = 0.32

    neg_items, pos_items = [], []
    for lab, val in zip(labels, top_vals):
        if val < 0:
            neg_items.append((lab, val))
        else:
            pos_items.append((lab, val))

    for lab, val in neg_items:
        width = abs(val)
        start = left_neg - width
        ax.barh(y_neg, width, left=start, height=bar_h, color="#1E88E5", edgecolor="white")
        ax.text(start + width / 2, y_neg, f"{val:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(start + width / 2, y_neg - 0.32, lab, ha="center", va="top", color="#1E88E5", fontsize=10)
        left_neg = start

    for lab, val in pos_items:
        width = abs(val)
        start = left_pos
        ax.barh(y_pos, width, left=start, height=bar_h, color="#FF0051", edgecolor="white")
        ax.text(start + width / 2, y_pos, f"+{val:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(start + width / 2, y_pos + 0.32, lab, ha="center", va="bottom", color="#FF0051", fontsize=10)
        left_pos = start + width

    ax.axvline(base_value, color="#888888", linestyle="--", linewidth=1.2)
    ax.axvline(pred_value, color="#222222", linestyle="-", linewidth=1.5)

    xmin = min(left_neg, base_value, pred_value) - 0.05
    xmax = max(left_pos, base_value, pred_value) + 0.05
    if xmin == xmax:
        xmin -= 0.1
        xmax += 0.1
    ax.set_xlim(xmin, xmax)

    ax.text(base_value, 0.72, f"Base value = {base_value:.3f}", ha="center", va="bottom", fontsize=11, color="#666666")
    ax.text(pred_value, 0.86, f"Prediction = {pred_value:.3f}", ha="center", va="bottom", fontsize=12, color="#111111", fontweight="bold")

    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Model output")
    ax.set_title("Force-style contribution plot", fontsize=12, fontweight="bold")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

    plt.tight_layout()
    return fig


def build_raw_input_df(ft_options):
    st.markdown(
        """
        <div style='background-color:orange;padding:4px 10px;border-radius:6px;display:inline-block;margin-top:18px;margin-bottom:18px;'>
            <h3 style='color:white;margin:0;'>Input parameters</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        pe = st.number_input("Pe", *RANGES["Pe"][:2], value=DEFAULT_INPUT["Pe"], step=RANGES["Pe"][2])
        du = st.number_input("Du", *RANGES["Du"][:2], value=DEFAULT_INPUT["Du"], step=RANGES["Du"][2])
        sp = st.number_input("SP", *RANGES["SP"][:2], value=DEFAULT_INPUT["SP"], step=RANGES["SP"][2])
        ac = st.number_input("AC", *RANGES["AC"][:2], value=DEFAULT_INPUT["AC"], step=RANGES["AC"][2])
        av = st.number_input("AV", *RANGES["AV"][:2], value=DEFAULT_INPUT["AV"], step=RANGES["AV"][2])

    with c2:
        vma = st.number_input("VMA", *RANGES["VMA"][:2], value=DEFAULT_INPUT["VMA"], step=RANGES["VMA"][2])
        vfa = st.number_input("VFA", *RANGES["VFA"][:2], value=DEFAULT_INPUT["VFA"], step=RANGES["VFA"][2])
        ag236 = st.number_input("Ag2.36", *RANGES["Ag2.36"][:2], value=DEFAULT_INPUT["Ag2.36"], step=RANGES["Ag2.36"][2])
        ag475 = st.number_input("Ag4.75", *RANGES["Ag4.75"][:2], value=DEFAULT_INPUT["Ag4.75"], step=RANGES["Ag4.75"][2])
        ag95 = st.number_input("Ag9.5", *RANGES["Ag9.5"][:2], value=DEFAULT_INPUT["Ag9.5"], step=RANGES["Ag9.5"][2])

    with c3:
        default_idx = ft_options.index(DEFAULT_INPUT["FT"]) if DEFAULT_INPUT["FT"] in ft_options else 0
        ft = st.selectbox("FT", options=ft_options, index=default_idx)
        fc = st.number_input("FC", *RANGES["FC"][:2], value=DEFAULT_INPUT["FC"], step=RANGES["FC"][2])
        fl = st.number_input("FL", *RANGES["FL"][:2], value=DEFAULT_INPUT["FL"], step=RANGES["FL"][2])
        ts = st.number_input("TS", *RANGES["TS"][:2], value=DEFAULT_INPUT["TS"], step=RANGES["TS"][2])

    raw_df = pd.DataFrame([{
        "Pe": pe,
        "Du": du,
        "SP": sp,
        "AC": ac,
        "AV": av,
        "VMA": vma,
        "VFA": vfa,
        "Ag2.36": ag236,
        "Ag4.75": ag475,
        "Ag9.5": ag95,
        "FT": ft,
        "FC": fc,
        "FL": fl,
        "TS": ts,
    }])
    return raw_df



# ============================================================
# 输入区
# ============================================================
ft_options = get_ft_options_from_preprocessor(artifacts["preprocessor"])
raw_input_df = build_raw_input_df(ft_options)

st.write("### Current raw input")
st.dataframe(raw_input_df, use_container_width=True)

# ============================================================
# 预测按钮
# ============================================================
predict_clicked = st.button("Predict ST and generate SHAP plots", use_container_width=True)

if predict_clicked:
    try:
        X_input = transform_input(artifacts["preprocessor"], raw_input_df)
        y_pred = safe_predict(artifacts["model"], X_input)

        st.session_state["raw_input_df"] = raw_input_df.copy()
        st.session_state["X_input"] = X_input.copy()
        st.session_state["y_pred"] = y_pred

        try:
            local_exp = make_local_shap_explanation(artifacts["explainer"], X_input)
            st.session_state["local_shap_exp"] = local_exp
            st.session_state["local_shap_ok"] = True
        except Exception:
            st.session_state["local_shap_ok"] = False
            st.session_state["local_shap_error"] = traceback.format_exc()

        st.success("Prediction finished.")

    except Exception:
        st.error("Prediction failed. See details below.")
        st.code(traceback.format_exc())

# ============================================================
# 预测结果区
# ============================================================
st.markdown(
    """
    <div style='background-color:orange;padding:4px 10px;border-radius:6px;display:inline-block;margin-top:18px;margin-bottom:18px;'>
        <h3 style='color:white;margin:0;'>Prediction result</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

if "y_pred" in st.session_state:
    st.markdown(
        f"""
        <div style="background-color:#F3F3F3;padding:14px;border-radius:8px;text-align:center;">
            <div style="font-size:28px;font-weight:800;color:#000000;line-height:1.4;">
                Predicted ST = {st.session_state['y_pred']:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show transformed model input"):
        st.dataframe(st.session_state["X_input"], use_container_width=True)
else:
    st.info("Click the prediction button to generate the ST result.")

# ============================================================
# 单样本 SHAP 解释区
# ============================================================
st.markdown(
    """
    <div style='background-color:orange;padding:4px 10px;border-radius:6px;display:inline-block;margin-top:18px;margin-bottom:18px;'>
        <h3 style='color:white;margin:0;'>SHAP analysis for the current sample</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

if "local_shap_ok" in st.session_state and st.session_state["local_shap_ok"]:
    local_exp = st.session_state["local_shap_exp"]
    sample_exp = local_exp[0]

    st.write("### Waterfall plot")
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        try:
            fig = plot_waterfall_from_explanation(sample_exp, max_display=12)
            st.pyplot(fig, clear_figure=True, use_container_width=False)
        except Exception:
            st.error("Failed to draw waterfall plot.")
            st.code(traceback.format_exc())

    st.write("### Contribution table")
    try:
        feature_names = list(sample_exp.feature_names)
        values = np.asarray(sample_exp.values).reshape(-1)
        feature_data = np.asarray(sample_exp.data).reshape(-1)

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Feature value": feature_data,
            "SHAP value": values,
            "abs(SHAP)": np.abs(values),
        }).sort_values("abs(SHAP)", ascending=False)
        st.dataframe(contrib_df, use_container_width=True)
    except Exception:
        st.error("Failed to build contribution table.")
        st.code(traceback.format_exc())

elif "local_shap_ok" in st.session_state and not st.session_state["local_shap_ok"]:
    st.warning("The current sample prediction succeeded, but local SHAP explanation could not be generated.")
    with st.expander("Show SHAP error details"):
        st.code(st.session_state.get("local_shap_error", "No details available."))
else:
    st.info("After prediction, the app will try to generate local SHAP plots if the explainer can be loaded.")
