import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hybrid Anomaly Detection", layout="wide")

# =========================
# Sidebar Configuration
# =========================

st.sidebar.title("Configuration")

dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ["NSL-KDD", "KDDCup"]
)

# =========================
# Load Models
# =========================

if dataset_choice == "KDDCup":
    svm_model = joblib.load("models/svm_kdd.pkl")
    lgb_model = joblib.load("models/lgb_kdd.pkl")
    top_features = joblib.load("models/features_kdd.pkl")
    scaler = joblib.load("models/scaler_kdd.pkl")
    feature_order = joblib.load("models/lgb_kdd_feature_order.pkl")
else:
    svm_model = joblib.load("models/svm_nsl.pkl")
    lgb_model = joblib.load("models/lgb_nsl.pkl")
    top_features = joblib.load("models/features_nsl.pkl")
    scaler = joblib.load("models/scaler_nsl.pkl")
    feature_order = joblib.load("models/lgb_nsl_feature_order.pkl")

# =========================
# Header Section
# =========================

st.markdown("""
    <div style='padding:25px;
                background: linear-gradient(90deg,#1f4bd8,#8f2eff);
                border-radius:15px;
                margin-bottom:25px'>
        <h1 style='color:white;'>Hybrid Anomaly Detection System</h1>
        <p style='color:white;'>Hybrid Explainable Network Intrusion Detection</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# File Upload Section
# =========================

st.subheader("File Analysis")

uploaded_file = st.file_uploader("Upload CSV for full prediction")

if uploaded_file is not None:

    input_data = pd.read_csv(uploaded_file)

    st.write("### Uploaded File Preview")
    st.dataframe(input_data.head())

    # =========================
    # SVM Prediction
    # =========================

    missing_features = set(top_features) - set(input_data.columns)

    if missing_features:
        st.error(f"Missing required selected feature columns: {missing_features}")
    else:
        X_input = input_data[top_features]
        X_scaled = scaler.transform(X_input)

        predictions = svm_model.predict(X_scaled)
        probabilities = svm_model.predict_proba(X_scaled)[:, 1]

        input_data["Prediction"] = predictions
        input_data["Attack Probability"] = probabilities

        # Risk Level
        def get_risk_level(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"

        input_data["Risk Level"] = input_data["Attack Probability"].apply(get_risk_level)

        st.success("Prediction completed successfully!")

        total = len(predictions)
        attacks = np.sum(predictions == 1)
        normal = np.sum(predictions == 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Attack Detected", attacks)
        col3.metric("Normal Traffic", normal)

        st.write("### Detailed Results")
        st.dataframe(input_data)

        # Download Button
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        # =========================
        # Prediction Distribution
        # =========================

        st.write("### Prediction Distribution")

        counts = input_data["Prediction"].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.bar(
            ["Normal", "Attack"],
            [counts.get(0, 0), counts.get(1, 0)]
        )
        ax1.set_title("Normal vs Attack Count")
        st.pyplot(fig1)

        # =========================
        # SHAP Explanation (Robust)
        # =========================

        explain = st.checkbox("Enable SHAP Explanation")

        if explain:

            st.write("### SHAP Feature Contribution Analysis")

            # Prepare input for LightGBM
            X_lgb = input_data.copy()
            X_lgb = X_lgb.drop(
                ["Prediction", "Attack Probability", "Risk Level"],
                axis=1,
                errors="ignore"
            )

            # Add missing columns
            for col in feature_order:
                if col not in X_lgb.columns:
                    X_lgb[col] = 0

            # Reorder columns
            X_lgb = X_lgb[feature_order]

            # SHAP
            explainer = shap.TreeExplainer(lgb_model)
            shap_values = explainer.shap_values(X_lgb.iloc[:1])

            shap_values = np.array(shap_values)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]

            shap_vector = shap_values[0]

            shap_df = pd.DataFrame({
                "Feature": feature_order,
                "Contribution": shap_vector
            }).sort_values(by="Contribution", key=abs, ascending=False)

            st.write("#### Top Feature Contributions")
            st.dataframe(shap_df.head(10))

            # Text Explanation
            prediction_label = "Attack" if predictions[0] == 1 else "Normal"

            positive = shap_df[shap_df["Contribution"] > 0]["Feature"].tolist()[:3]
            negative = shap_df[shap_df["Contribution"] < 0]["Feature"].tolist()[:3]

            st.write("### Textual Explanation")

            if prediction_label == "Attack":
                explanation = f"""
                The model classified this traffic as **Attack** because
                {', '.join(positive)} significantly increased the attack probability.
                """
            else:
                explanation = f"""
                The model classified this traffic as **Normal** because
                {', '.join(negative)} significantly reduced the attack probability.
                """

            st.info(explanation)

# =========================
# Manual Input Section
# =========================

st.subheader("Manual Single Record Prediction")

manual_data = {}
cols = st.columns(3)

for i, feature in enumerate(top_features):
    manual_data[feature] = cols[i % 3].number_input(
        feature,
        value=0.0
    )

if st.button("Predict Manual Input"):

    input_df = pd.DataFrame([manual_data])
    X_scaled = scaler.transform(input_df)

    prediction = svm_model.predict(X_scaled)[0]
    probability = svm_model.predict_proba(X_scaled)[0][1]

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    st.write("### Manual Prediction Result")
    st.write("Prediction:", "Attack" if prediction == 1 else "Normal")
    st.write("Attack Probability:", round(probability, 4))
    st.write("Risk Level:", risk)