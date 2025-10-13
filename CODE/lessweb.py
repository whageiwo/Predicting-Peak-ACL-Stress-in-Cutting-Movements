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
feature_full_names = [
    "Hip Flexion Angle(HFA)", "Knee Flexion Angle(KFA)", "Hip Adduction Ankle(HAA)",
    "Knee Valgus Ankle(KVA)", "Ankle Valgus Ankle(AVA)", "Knee Valgus Moment(KVM)",
    "Knee Flexion moment(KFM)", "Anterior Tibial Shear Force (ASF)", "Hamstring/Quadriceps(H/Q)"
]

feature_abbreviations = ["HFA", "KFA", "HAA", "KVA", "AVA", "KVM", "KFM", "ASF", "H/Q"]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# -------- 输入特征值 --------
with col1:
    for name in feature_full_names[:5]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

with col2:
    for name in feature_full_names[5:]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 预测结果 --------
pred = model.predict(X_input)[0]

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:darkgreen;'>Predicted Value</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>{pred:.3f}</p>", unsafe_allow_html=True)

# -------- SHAP 可视化 --------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)
    
    # 确保SHAP值与特征数量匹配
    if len(shap_values.values[0]) != len(feature_abbreviations):
        st.error(f"特征数量不匹配！SHAP值数量：{len(shap_values.values[0])}，特征数量：{len(feature_abbreviations)}")
    else:
        # 使用更稳定的API创建Explanation对象
        try:
            # 方法1：使用values数组直接创建
            shap_expl = shap.Explanation(
                values=shap_values.values[0],
                base_values=shap_values.base_values[0],
                data=X_input[0],
                feature_names=feature_abbreviations
            )
            
            # --- 瀑布图 ---
            st.markdown("<h3 style='color:darkorange;'>Waterfall Plot</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 使用更稳定的绘图方式
            try:
                shap.plots.waterfall(shap_expl, show=False)
            except Exception as e:
                st.warning(f"瀑布图错误: {str(e)}")
                # 备用方案：使用条形图
                shap.plots.bar(shap_expl, show=False)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"创建Explanation对象错误: {str(e)}")
            # 方法2：直接使用shap_values[0]
            st.markdown("<h3 style='color:darkorange;'>Waterfall Plot (Alternative)</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            shap.plots.waterfall(shap_values[0], show=False, feature_names=feature_abbreviations)
            st.pyplot(fig)

# ------------------ 力图 ------------------
st.markdown("<h3 style='color:purple; text-align:center;'>Force Plot</h3>", unsafe_allow_html=True)

try:
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_abbreviations,
        plot_cmap="PkYg",
        contribution_threshold=0.001
    )
    
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
    </style>
    <div class="scroll-container">
        <head>{shap.getjs()}</head>
        {force_plot.html()}
    </div>
    """
    components.html(html_code, height=400, scrolling=True)
    
except Exception as e:
    st.error(f"创建力图错误: {str(e)}")
    # 显示原始SHAP值作为备用
    with st.expander("SHAP Values"):
        st.write(shap_values.values)

