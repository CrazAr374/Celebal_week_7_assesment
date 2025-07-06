import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Load model and features
model, feature_names = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="🏦 Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.markdown("Provide applicant details to predict loan approval using a trained ML model.")

# Input UI
def user_input():
    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Married", ['Yes', 'No'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    applicant_income = st.slider("Applicant Income", 0, 100000, 5000)
    coapplicant_income = st.slider("Coapplicant Income", 0, 50000, 0)
    loan_amount = st.slider("Loan Amount", 10, 700, 150)
    loan_amount_term = st.slider("Loan Term (in days)", 12, 480, 360)
    property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

    # Prepare input data
    data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Married_Yes': 1 if married == 'Yes' else 0,
        'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
        'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
        'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
    }

    df = pd.DataFrame([data])
    return df

input_df = user_input()

# Ensure all features exist
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Prediction: {'Approved ✅' if prediction == 1 else 'Not Approved ❌'}")
    st.write(f"Confidence Score: **{round(prob * 100, 2)}%**")

    # SHAP Explanation
    st.subheader("📈 Model Explainability (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.summary_plot(shap_values[1], input_df, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
