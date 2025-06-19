import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load preprocessor and model using pickle
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("💓 Heart Attack Risk Prediction")
st.markdown("Predict your heart attack risk based on your health and lifestyle indicators.")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])  # categorical
    cholesterol = st.number_input("Cholesterol (mg/dL)", value=190.0)

    bp_input = st.text_input("Blood Pressure (e.g., 120/80)", value="120/80")
    try:
        bp_up, bp_down = map(float, bp_input.strip().split('/'))
    except:
        st.error("Please enter blood pressure in the format '120/80'")
        st.stop()

    heart_rate = st.number_input("Heart Rate (bpm)", value=72)

    diabetes = st.selectbox("Do you have Diabetes?", ["No", "Yes"])
    diabetes = 1 if diabetes == "Yes" else 0

    family_history = st.selectbox("Family History of Heart Issues?", ["No", "Yes"])
    family_history = 1 if family_history == "Yes" else 0

    smoking = st.selectbox("Do you Smoke?", ["No", "Yes"])
    smoking = 1 if smoking == "Yes" else 0

    obesity = st.selectbox("Are you Obese?", ["No", "Yes"])
    obesity = 1 if obesity == "Yes" else 0

    alcohol = st.selectbox("Do you consume Alcohol?", ["No", "Yes"])
    alcohol = 1 if alcohol == "Yes" else 0

    exercise = st.slider("Exercise Hours Per Week", 0, 20, 3)

    diet = st.selectbox("Diet", ["Healthy", "Unhealthy"])  # categorical

    previous_problems = st.selectbox("Any Previous Heart Problems?", ["No", "Yes"])
    previous_problems = 1 if previous_problems == "Yes" else 0

    medication = st.selectbox("Do you use Medication?", ["No", "Yes"])
    medication = 1 if medication == "Yes" else 0

    stress = st.slider("Stress Level (0 - Low, 10 - High)", 0, 10, 5)
    sedentary = st.slider("Sedentary Hours Per Day", 0, 24, 8)
    income = st.number_input("Annual Income ($)", value=50000.0)
    bmi = st.number_input("BMI", value=24.0)
    triglycerides = st.number_input("Triglycerides (mg/dL)", value=150.0)
    activity_days = st.slider("Physical Activity Days Per Week", 0, 7, 3)
    sleep = st.slider("Sleep Hours Per Day", 0, 24, 7)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],  # categorical
        'Cholesterol': [cholesterol],
        'Heart Rate': [heart_rate],
        'Diabetes': [diabetes],
        'Family History': [family_history],
        'Smoking': [smoking],
        'Obesity': [obesity],
        'Alcohol Consumption': [alcohol],
        'Exercise Hours Per Week': [exercise],
        'Diet': [diet],  # categorical
        'Previous Heart Problems': [previous_problems],
        'Medication Use': [medication],
        'Stress Level': [stress],
        'Sedentary Hours Per Day': [sedentary],
        'Income': [income],
        'BMI': [bmi],
        'Triglycerides': [triglycerides],
        'Physical Activity Days Per Week': [activity_days],
        'Sleep Hours Per Day': [sleep],
        'up': [bp_up],
        'down': [bp_down]
    })

    # Transform input
    transformed_input = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(transformed_input)[0]
    probability = model.predict_proba(transformed_input)[0][1]

    # Output
    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack Detected.\n\n**Probability: {probability:.2%}**")
    else:
        st.success(f"✅ Low Risk of Heart Attack.\n\n**Probability: {probability:.2%}**")
