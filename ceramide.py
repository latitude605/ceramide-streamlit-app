
import streamlit as st
import shap
import numpy as np
import pandas as pd
import pickle
import streamlit.components.v1 as components
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import os

feature_names = ["Sex", "HTN", "DM", "CVA", "HDL.C", "LDL.C", "Cer16:0", "Cer24:1"]


# 创建用户输入特征字段，不指定范围
input_data = {}
default_value = 0.00  # 这里可以根据需要调整默认值
for feature in feature_names:
    input_data[feature] = st.number_input(f'{feature}', value=default_value)

# 创建输入样本
user_input = np.array([list(input_data.values())])


    # 加载模型
        with open('model.pkl', 'rb') as file:
            model, X, y = pickle.load(file)
            
  # 创建 Streamlit 应用程序
st.title("SHAP Analysis")



if st.button('Predict'):
    try:
        prediction = model.predict(user_input)[0]
        explainer = shap.Explainer(model.predict_proba, X, feature_names=feature_names)
        shap_values = explainer(user_input)

        st.write(prediction)
        
        shap.initjs()
        shap_force_plot = shap.plots.force(shap_values[:,:,1])
        
        shap_force_html = f"<head>{shap.getjs()}</head><body>{shap_force_plot.html()}</body>"
        components.html(shap_force_html, height=1000)
    except Exception as e:
        st.error(f"Error: {e}")
