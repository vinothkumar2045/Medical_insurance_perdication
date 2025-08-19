import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load(r"C:\Users\cpvin\OneDrive\Documents\Guvi_mini_projects\Medical Insurance Cost Prediction\best_model.pkl")
scaler = joblib.load(r"C:\Users\cpvin\OneDrive\Documents\Guvi_mini_projects\Medical Insurance Cost Prediction\caler.pkl")

st.title("🏥 Medical Insurance Cost Prediction")
st.write("Enter patient details to predict insurance charges.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input for prediction
input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "Male" else 0,
    "smoker_yes": 1 if smoker == "Yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}

# Order features correctly
feature_order = ["age", "bmi", "children", "sex_male", "smoker_yes",
                 "region_northwest", "region_southeast", "region_southwest"]

input_array = np.array([input_data[feature] for feature in feature_order]).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_array)

# Prediction
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Insurance Cost: ${prediction:.2f}")
