import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 
# PAGE CONFIG
# 
st.set_page_config(
    page_title="Hybrid Explainable Anomaly Detection",
    layout="wide"
)

st.title("🔐 Hybrid Explainable Network Anomaly Detection System")
st.markdown("LightGBM (Feature Selection) + SVM (Classification) + SHAP (Explainability)")

# 
# MODEL SELECTION
# 
dataset_option = st.sidebar.selectbox(
    "Select Dataset Model",
    ["KDDCup99 Model", "NSL-KDD Model"]
)

# 
# LOAD MODELS BASED ON SELECTION
# 
if dataset_option == "KDDCup99 Model":
    svm_model = joblib.load("saved_models/svm_kdd.pkl")
    lgb_model = joblib.load("saved_models/lgb_kdd.pkl")
    scaler = joblib.load("saved_models/scaler_kdd.pkl")
    selected_features = joblib.load("saved_models/selected_features_kdd.pkl")
else:
    svm_model = joblib.load("saved_models/svm_nsl.pkl")
    lgb_model = joblib.load("saved_models/lgb_nsl.pkl")
    scaler = joblib.load("saved_models/scaler_nsl.pkl")
    selected_features = joblib.load("saved_models/selected_features_nsl.pkl")

# 
# INPUT SECTION
# 
st.subheader("📥 Enter Feature Values")

input_data = []

cols = st.columns(3)

for i, feature in enumerate(selected_features):
    with cols[i % 3]:
        value = st.number_input(f"{feature}", value=0.0)
        input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

# 
# PREDICTION BUTTON
# 
if st.button("🚀 Predict"):
    
    # Scale input
    scaled_input = scaler.transform(input_array)

    # Predict
    prediction = svm_model.predict(scaled_input)
    probability = svm_model.predict_proba(scaled_input)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ Attack Detected")
    else:
        st.success("✅ Normal Traffic")

    st.write(f"Confidence Score: {probability:.4f}")

    # 
    # PROBABILITY BAR VISUALIZATION
    # 
    st.subheader("📈 Prediction Probability")

    prob_df = pd.DataFrame({
        "Class": ["Normal", "Attack"],
        "Probability": [1 - probability, probability]
    })

    st.bar_chart(prob_df.set_index("Class"))

    # 
    # SHAP EXPLANATION
    # 
    st.subheader("🔍 SHAP Feature Contribution")

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(scaled_input)

    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        scaled_input,
        feature_names=selected_features,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
