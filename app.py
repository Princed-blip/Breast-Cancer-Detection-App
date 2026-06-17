import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image



# Load model and scaler
model = pickle.load(open('BCC_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .title {
        text-align: center;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)



# Image
# st.image("Breast_cancer.png", use_container_width=True)
img = Image.open("Image.png")
img = img.resize((800, 250))  # width, height
st.image(img)
    


# # Title
# st.markdown("<h1 style='color: skyblue; font-family: Times New Roman;' class='title'>BREAST CANCER PREDICTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("---")



# Sidebar
st.sidebar.markdown("<h1 style='color: skyblue; font-family: Segoe Script;'>⚕️Biomedical Engineering, UNN</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")


st.sidebar.title("About This Project")

with st.sidebar.expander("Click to view"):
    st.info("""
This application is an AI-powered Breast Cancer Prediction System designed to assist in the early detection of breast cancer using machine learning techniques.

The project began with a detailed Exploratory Data Analysis (EDA) on a clinical dataset containing 569 patient records and multiple tumor-related features such as radius, texture, smoothness, concavity, and symmetry. Data preprocessing steps were performed to improve data quality, including handling skewness, detecting and treating outliers, and scaling numerical features.

To reduce redundancy and improve model performance, highly correlated features were identified and removed, while the most informative features were retained based on their relationship with the diagnosis (Benign or Malignant).

Several machine learning algorithms were trained and evaluated, including Logistic Regression, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). Logistic Regression was selected as the final model due to its strong performance, achieving over 96% accuracy and a high recall score, which is especially important in medical diagnosis to minimize missed cancer cases.

The model was further evaluated using metrics such as confusion matrix, ROC curve, and AUC score, confirming its excellent ability to distinguish between benign and malignant tumors.

This interactive application allows users to input tumor characteristics and receive real-time predictions along with probability scores, demonstrating how AI can support decision-making in healthcare.

⚠️ Disclaimer: This tool is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.
"""
)

# Adding how to use
st.sidebar.title("How to Use")
with st.sidebar.expander("Click to view"):
    st.info("""
1. Enter the tumor characteristics in the input fields provided (e.g., radius, texture, concavity, etc.).

2. Ensure all values are filled correctly based on the patient’s measurements.

3. Click the **“Predict”** button to analyze the data.

4. The system will display:
   - The predicted diagnosis (**Benign or Malignant**)
   - The probability score indicating confidence level

5. Interpret the result:
   - **Benign (Low Risk)** → Non-cancerous tumor
   - **Malignant (High Risk)** → Cancer detected

6. Use the feature importance chart to understand which features influenced the prediction.
"""
)
    

# Layout columns
col1, col2, col3 = st.columns(3)

# -------- COLUMN 1 --------
with col1:
    st.subheader("📏 Mean Features")
    radius_mean = st.number_input("Radius Mean", value=10.0)
    texture_mean = st.number_input("Texture Mean", value=20.0)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.1)
    compactness_mean = st.number_input("Compactness Mean", value=0.1)
    concavity_mean = st.number_input("Concavity Mean", value=0.1)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.05)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.2)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)

# -------- COLUMN 2 --------
with col2:
    st.subheader("📊 Standard Error Features")
    radius_se = st.number_input("Radius SE", value=0.5)
    texture_se = st.number_input("Texture SE", value=1.0)
    smoothness_se = st.number_input("Smoothness SE", value=0.01)
    compactness_se = st.number_input("Compactness SE", value=0.02)
    concavity_se = st.number_input("Concavity SE", value=0.02)
    concave_points_se = st.number_input("Concave Points SE", value=0.01)
    symmetry_se = st.number_input("Symmetry SE", value=0.02)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.001)

# -------- COLUMN 3 --------
with col3:
    st.subheader("📈 Worst Features")
    radius_worst = st.number_input("Radius Worst", value=15.0)
    texture_worst = st.number_input("Texture Worst", value=25.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.1)
    compactness_worst = st.number_input("Compactness Worst", value=0.2)
    concavity_worst = st.number_input("Concavity Worst", value=0.2)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.1)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.3)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)

# st.markdown("---")

# Prediction
if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array([[
        radius_mean, texture_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean,
        symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
        smoothness_se, compactness_se, concavity_se, concave_points_se,
        symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
        smoothness_worst, compactness_worst, concavity_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("##  Result")

    if prediction[0] == 1:
        st.error(f"Malignant (High Risk, Cancerous!)\n\nProbability: {probability:.2f}")
    else:
        st.success(f"Benign (Low Risk, Non Cancerous)\n\nProbability: {probability:.2f}")

    st.progress(float(probability))


# Feature Importance
# Feature Importance
with st.expander("📊 Feature Importance (Click to view)"):
    feature_names = [
        'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 
        'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
        'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
        'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 
        'texture_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    importance = model.coef_[0]
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_imp = feat_imp.reindex(feat_imp.Importance.abs().sort_values(ascending=False).index).head(10)
    
    plt.figure(figsize=(8,6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Logistic Regression)")
    plt.gca().invert_yaxis()
    st.pyplot(plt)


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>© 2026 | Developed by PRINCEDEX</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Contact: [ifeanyistephen003@gmail.com]</p>",
    unsafe_allow_html=True
)

    