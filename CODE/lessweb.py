import streamlit as st
import xgboost as xgb
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="Predicting Peak ACL Stress in Cutting Movements", layout="wide")

# ------------------ 全局字体 ------------------
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred;'>Predicting Peak ACL Stress in Cutting Movements</h1>", unsafe_allow_html=True)

# ------------------ 加载模型 ------------------
model = joblib.load("final_XGJ_model.bin")

# ------------------ 定义特征名称 ------------------
# 原始特征名称（用于输入和预测）
feature_names = [
    "Hip Flexion Angle(HFA)", "Knee Flexion Angle(KFA)", "Hip Adduction Ankle(HAA)",
    "Knee Valgus Ankle(KVA)", "Ankle Valgus Ankle(AVA)", "Knee Valgus Moment(KVM)",
    "Knee Flexion moment(KFM)", "Anterior Tibial Shear Force (ASF)", "Hamstring/Quadriceps(H/Q)"
]

# 简写特征名称（用于可视化）
short_feature_names = [
    "Hip Flexion", "Knee Flexion", "Hip Adduction",
    "Knee Valgus", "Ankle Valgus", "KVM",
    "KFM", "Anterior Shear", "H/Q Ratio"
]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# -------- 左列前5个特征 --------
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# -------- 中列后4个特征 --------
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

# -------- 右列：SHAP 可视化 --------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)
    
    # --- 瀑布图（使用简写特征名称）---
    st.markdown("<h3 style='color:darkorange;'>Waterfall Plot</h3>", unsafe_allow_html=True)
    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=short_feature_names  # 使用简写名称
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    shap.plots.waterfall(shap_expl, show=False)
    st.pyplot(fig)
    
    # --- 力图（使用简写特征名称）---
    st.markdown("<h3 style='color:purple;'>Force Plot</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=short_feature_names  # 使用简写名称
    )
    
    # 创建带滚动条的容器
    html_code = f"""
    <style>
    .shap-force-plot text {{
        font-size: 12px !important;
    }}
    .scroll-container {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        background-color: white;
    }}
    .scroll-container::-webkit-scrollbar {{
        height: 12px;
    }}
    .scroll-container::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 6px;
    }}
    .scroll-container::-webkit-scrollbar-thumb {{
        background: #888;
        border-radius: 6px;
    }}
    .scroll-container::-webkit-scrollbar-thumb:hover {{
        background: #555;
    }}
    </style>
    <div class="scroll-container">
        <head>{shap.getjs()}</head>
        {force_plot.html()}
    </div>
    """
    components.html(html_code, height=300)


