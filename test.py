import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess

st.set_page_config(page_title="Medical Insurance Cost Estimator", layout="wide")

# Custom CSS for background and styling
st.markdown("""
    <style>
      
        /* Centered title */
        h1 {
            text-align: center;
            color: #002b5b;
        }

        h2, h3 {
            color: #005792;
        }
        
     </style>
     

""", unsafe_allow_html=True)
import streamlit as st

# Full-page background image with overlay for readability
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-image: url("C:/Users/cpvin/Downloads/images.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Optional overlay (makes content easier to read) */
        .main > div {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        h1, h2, h3 {
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load(r"C:/Users/cpvin/Downloads/Copy of best_model.csv")

# Title
st.title("💼 Medical Insurance Cost Estimator")

# Sidebar
st.sidebar.title("📌 Navigation")
nav_choice = st.sidebar.radio("Go to", ["EDA Insights", "Predict Insurance Cost"])

# Load and preprocess once
df = load_and_preprocess()

# Main container
with st.container():
    if nav_choice == "EDA Insights":
        st.header("📊 Exploratory Data Analysis")

        eda_option = st.sidebar.selectbox("Select EDA Chart", [
            "Distribution of Medical Charges",
            "Charges vs Age by Smoking Status",
            "Correlation Heatmap"
        ])

        if eda_option == "Distribution of Medical Charges":
            st.subheader("Distribution of Medical Charges")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['charges'], kde=True, ax=ax1, color="skyblue")
            st.pyplot(fig1)

        elif eda_option == "Charges vs Age by Smoking Status":
            st.subheader("Charges vs Age by Smoking Status")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6, ax=ax2)
            st.pyplot(fig2)

        elif eda_option == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            fig3, ax3 = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax3)
            st.pyplot(fig3)

    elif nav_choice == "Predict Insurance Cost":
        st.header("💡 Predict Your Insurance Cost")

        age = st.slider("Age", 18, 65, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.slider("Number of Children", 0, 5, 0)
        smoker = st.selectbox("Do you Smoke?", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        # Encode input
        sex = 1 if sex == "male" else 0
        smoker = 1 if smoker == "yes" else 0
        region_dict = {"northeast": 3, "northwest": 2, "southeast": 1, "southwest": 0}
        region = region_dict[region]

        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                  columns=["age", "sex", "bmi", "children", "smoker", "region"])

        if st.button("Predict Insurance Cost"):
            prediction = model.predict(input_data)[0]
            st.success(f"💲 Estimated Insurance Charges: ${prediction:,.2f}")
