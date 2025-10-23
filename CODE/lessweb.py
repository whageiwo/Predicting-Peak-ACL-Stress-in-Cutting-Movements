import streamlit as st
import xgboost as xgb
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(
    page_title="Predicting Peak ACL Stress in Cutting Movements",
    layout="wide"
)

# ------------------ 全局字体 ------------------
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# ------------------ 页面标题 ------------------
st.markdown(
    "<h1 style='text-align: center; color: darkred;'>Predicting Peak ACL Stress in Cutting Movements</h1>",
    unsafe_allow_html=True
)

# ------------------ 版本信息 ------------------
st.sidebar.markdown("### Environment Info")
st.sidebar.write(f"**SHAP version:** {shap.__version__}")
st.sidebar.write(f"**XGBoost version:** {xgb.__version__}")

# ------------------ 加载模型 ------------------
model = joblib.load("final_XGJ_model.bin")  # ✅ joblib加载

# ------------------ 定义特征名称 ------------------
feature_names = [
    "Hip Flexion Angle(HFA)",
    "Knee Flexion Angle(KFA)",
    "Hip Adduction Angle(HAA)",
    "Knee Valgus Angle(KVA)",
    "Ankle Valgus Angle(AVA)",
    "Knee Valgus Moment(KVM)",
    "Knee Flexion moment(KFM)",
    "Anterior Tibial Shear Force(ASF)",
    "Hamstring/Quadriceps(H/Q)"
]

# 特征缩写（用于 SHAP 可视化）
feature_short_names = ["HFA", "KFA", "HAA", "KVA", "AVA", "KVM", "KFM", "ASF", "H/Q"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# -------- 左列前 5 个特征 --------
with col1:
    for name in feature_names[:5]:
        st.markdown(
            f"<p style='font-size:{label_size}; margin:0'>{name}</p>",
            unsafe_allow_html=True
        )
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# -------- 中列后 4 个特征 --------
with col2:
    for name in feature_names[5:]:
        st.markdown(
            f"<p style='font-size:{label_size}; margin:0'>{name}</p>",
            unsafe_allow_html=True
        )
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 回归预测输出 --------
pred = model.predict(X_input)[0]

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='color:darkgreen;'>Predicted Value</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color:blue; font-size:40px; font-weight:bold;'>{pred:.3f}</p>",
        unsafe_allow_html=True
    )

# -------- 右列：SHAP 可视化（瀑布图 + 力图） --------
with col3:
    try:
        # ✅ 优先使用新版 SHAP 的通用解释器
        explainer = shap.Explainer(model)
        shap_values = explainer(X_input)

        # --- 创建 Explanation 对象 ---
        shap_expl = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=X_input[0],
            feature_names=feature_short_names
        )

        # --- 瀑布图 ---
        st.markdown(
            "<h3 style='color:darkorange;'>Waterfall Plot</h3>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.plots.waterfall(shap_expl, show=False)
        st.pyplot(fig)

        # --- 力图 ---
        st.markdown(
            "<h3 style='color:purple;'>Force Plot</h3>",
            unsafe_allow_html=True
        )
        force_plot = shap.force_plot(
            shap_expl.base_values,
            shap_expl.values,
            shap_expl.data,
            feature_names=feature_short_names
        )
        components.html(
            f"<head>{shap.getjs()}</head>{force_plot.html()}",
            height=300
        )

    except Exception as e:
        # ✅ 兼容旧版 SHAP（如0.41~0.45）
        st.warning("⚠️ Detected SHAP/XGBoost version incompatibility, using fallback mode.")
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_input)

        shap_expl = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_input[0],
            feature_names=feature_short_names
        )

        st.markdown(
            "<h3 style='color:darkorange;'>Waterfall Plot (Fallback)</h3>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        shap.plots.waterfall(shap_expl, show=False)
        st.pyplot(fig)

        st.error(f"Error: {str(e)}")


