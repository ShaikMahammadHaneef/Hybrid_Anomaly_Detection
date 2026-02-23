# Hybrid Explainable Network Intrusion Detection System (IDS)

A hybrid machine learning framework for detecting network intrusions using LightGBM-based feature selection, SVM classification, and SHAP explainability.

---

## Overview

This project implements a robust and interpretable Intrusion Detection System (IDS) evaluated on benchmark datasets. The system combines feature importance estimation, hybrid classification, and explainable AI to improve detection reliability and transparency.

---

## Datasets Used

- **KDDCup 1999**
- **NSL-KDD (Train & Test)**

Both datasets contain 41 network traffic features representing normal and attack behavior.

---

## Methodology

Data Preprocessing  
→ Feature Selection (LightGBM)  
→ Hybrid Classification (SVM – RBF Kernel)  
→ SHAP Explainability  
→ Evaluation & Visualization  
→ Streamlit Deployment  

---

## Model Architecture

- **LightGBM**
  - Used for feature importance estimation
  - Reduces dimensionality by selecting top features

- **Support Vector Machine (SVM)**
  - Final classifier
  - RBF kernel with class balancing

- **SHAP (Explainable AI)**
  - Provides feature-level contribution
  - Generates textual explanation for predictions

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

---

## Deployment

A Streamlit web application allows:

- CSV upload for batch prediction  
- Attack probability estimation  
- Risk level classification  
- SHAP-based feature contribution visualization  
- Prediction distribution graph  

Run locally:

-bash
streamlit run webapp.py
## Tech Stack

Python
LightGBM
Scikit-learn
SHAP
Pandas / NumPy
Matplotlib / Seaborn
Streamlit
