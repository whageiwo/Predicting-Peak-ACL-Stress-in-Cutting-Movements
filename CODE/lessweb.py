import streamlit as st
import xgboost as xgb
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="Predicting Peak ACL Stress in Cutting Movements", layout="wide")

# ------------------ 全局字体 ------------------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']  # 中文兼容
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# ------------------ 页面标题 ------------------
st.markdown(
    "<h1 style='text-align: center; color: darkred;'>Predicting Peak ACL Stress in Cutting Movements</h1>",
    unsafe_allow_html=True
)

# ------------------ 加载模型 ------------------
model_path = os.path.join(os.path.dirname(__file__), "final_XGJ_model.bin")
model = joblib.load(model_path)

# ------------------ 定义特征名称 ------------------
feature_names = [
    "Hip Flexion Angle(HFA)", "Knee Flexion Angle(KFA)", "Hip Adduction Angle(HAA)",
    "Knee Valgus Angle(KVA)", "Ankle Valgus Angle(AVA)", "Knee Valgus Moment(KVM)",
    "Knee Flexion Moment(KFM)", "Anterior Tibial Shear Force(ASF)", "Hamstring/Quadriceps(H/Q)"
]

# 对应的缩写，用于 SHAP 可视化
feature_short_names = ["HFA", "KFA", "HAA", "KVA", "AVA", "KVM", "KFM", "ASF", "H/Q"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# -------- 左列前 5 个特征 --------
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# -------- 中列后 4 个特征 --------
with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 回归预测输出 --------
pred = model.predict(X_input)[0]

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:darkgreen;'>Predicted Value</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>{pred:.3f}</p>", unsafe_allow_html=True)

# -------- 右列：SHAP 可视化（瀑布图 + 力图） --------
with col3:
    # 初始化 SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    # 构建 Explanation 对象
    shap_expl = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_input[0],
        feature_names=feature_short_names
    )

    # --- 瀑布图 ---
    st.markdown("<h3 style='color:darkorange;'>Waterfall Plot</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    shap.plots.waterfall(shap_expl, show=False)
    st.pyplot(fig)
    plt.close(fig)

    # --- 力图 ---
    st.markdown("<h3 style='color:purple;'>Force Plot</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_input[0],
        feature_names=feature_short_names,
        matplotlib=False
    )
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    components.html(force_html, height=300)
