#!/usr/bin/env python
# coding: utf-8

# In[19]:


import streamlit as st

import joblib

import pandas as pd

import numpy as np

import shap

import matplotlib.pyplot as plt


# In[21]:


model = joblib.load('lgb.pkl')


# In[23]:


X_test = pd.read_csv('X_test_HTN.csv')


# In[25]:


#定义特征名称，对应数据集中的列名
feature_names = ["YiDC", "PDC", "Age", "WHtR", "CO", "BMI", "Smokingstatus","Socialparticipation","Sleepquality","Pain","Hyperlipidemia", "Hyperuricemia","Diabetes","CKD"]


# In[27]:

# 构建字段映射
BOOL = {"Yes":1, "No":0}
AGE = {"60-69":0, "70-79":1, "≥80":2}
BMIV = {"18.5≤BMI<24":0, "<18.5":1, "24≤BMI<28":2, "≥28":3}

#Streamlit 用户界面
st.title("Hypertension Risk Prediction")
YiDC = BOOL[st.selectbox("Yin-deficiency constitution (YiDC):", options=BOOL)]
PDC = BOOL[st.selectbox("Phlegm-dampness constitution (PDC):", options=BOOL)]
Age = AGE[st.selectbox("Age:", options=AGE)]
WHtR = BOOL[st.selectbox("Waist-to-height ratio (WHtR):", options=BOOL)]
CO = BOOL[st.selectbox("Central obesity (CO):", options=BOOL)]
BMI = BMIV[st.selectbox("Body Mass Index (BMI):", options=BMIV)]
Smokingstatus = BOOL[st.selectbox("Smokingstatus:", options=BOOL)]
Socialparticipation = BOOL[st.selectbox("Socialparticipation:", options=BOOL)]
Sleepquality = BOOL[st.selectbox("Sleepquality:", options=BOOL)]  # 修改点: 之前缺失了
Pain = BOOL[st.selectbox("Pain:", options=BOOL)]
Hyperlipidemia = BOOL[st.selectbox("Hyperlipidemia:", options=BOOL)]
Hyperuricemia = BOOL[st.selectbox("Hyperuricemia:", options=BOOL)]
Diabetes = BOOL[st.selectbox("Diabetes:", options=BOOL)]
CKD = BOOL[st.selectbox("CKD:", options=BOOL)]


# In[29]:


# 实现输入数据并进行预测
feature_values = [YiDC, PDC, Age, WHtR, CO, BMI, Smokingstatus, Socialparticipation, Sleepquality, Pain, Hyperlipidemia, Hyperuricemia, Diabetes, CKD]  # 将用户输入的特征值存入列表   # 修改点: Sleepquality

features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入
# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0: 无高血压，1: 有高血压）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 修改点
    explainer_shap.expected_value = [explainer_shap.expected_value]

    # 根据预测结果生成建议
    # 如果预测类别为 1（高风险）
    if predicted_class==1: # float(predicted_proba[1])>explainer_shap.expected_value[1]:  # 修改点，不能这样判断 float(predicted_proba[1])>explainer_shap.expected_value[1]
        probability = predicted_proba[1] * 100
        advice = (
            f"According to our model, you have a high risk of hypertension. "
            f"The model predicts that your probability of having hypertension is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为 0（低风险）
    else:
        probability = predicted_proba[0] * 100
        advice = (
            f"According to our model, you have a low risk of hypertension. "
            f"The model predicts that your probability of not having hypertension is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    
    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    shap.force_plot(explainer_shap.expected_value[1], shap_values[1][0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)  # 修改点: shap_values[1][0]
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图 
    #plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.pyplot(plt.gcf(), use_container_width=True)


# In[ ]:




