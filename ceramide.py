
import streamlit as st
import shap
import numpy as np
import pandas as pd
import pickle
import streamlit.components.v1 as components
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 定义特征名称
feature_names = ["Sex", "HTN", "DM", "CVA", "HDL.C", "LDL.C", "Cer16:0", "Cer24:1"]

def main():
    st.title("Ceramide Diagnostic Model for Unstable Angina")
    
    # 创建用户输入特征字段
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f'{feature}', value=0.00, format="%.2f")
    
    # 加载模型
    try:
        with open('model.pkl', 'rb') as file:
            model, X, y = pickle.load(file)
    except Exception as e:
        st.error(f"模型加载错误: {e}")
        return
    
    # 预测按钮
    if st.button('预测'):
        try:
            # 创建输入样本
            user_input = np.array([list(input_data.values())])
            
            # 进行预测
            prediction = model.predict(user_input)[0]
            
            # SHAP解释
            explainer = shap.Explainer(model.predict_proba, X, feature_names=feature_names)
            shap_values = explainer(user_input)
            
            # 展示预测结果
            st.write(f"预测结果: {prediction}")
            
            # 显示SHAP力图
            shap.initjs()
            shap_force_plot = shap.plots.force(shap_values[:,:,1])
            shap_force_html = f"<head>{shap.getjs()}</head><body>{shap_force_plot.html()}</body>"
            components.html(shap_force_html, height=1000)
        
        except Exception as e:
            st.error(f"预测错误: {e}")

if __name__ == "__main__":
    main()
