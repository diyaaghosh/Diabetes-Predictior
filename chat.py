import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load objects
model = joblib.load("diabetes_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction",page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below.")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 1000, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    # Same columns used during training
    cols_with_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']

    # Replace zero with NaN
    input_df[cols_with_zero_missing] = (input_df[cols_with_zero_missing].replace(0, np.nan))

    # Impute only these columns
    input_df[cols_with_zero_missing] = imputer.transform(input_df[cols_with_zero_missing])

    # Scale all features
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if probability < 0.3:
        st.success(f"🟢 Low Diabetes Risk ({probability*100:.2f}%)")

        st.info("""
            Recommended actions:
            - Continue a balanced diet.
            - Exercise regularly.
            - Maintain healthy body weight.
            - Perform routine health checkups.
            """)
    else:
        st.error(f"🔴 High Diabetes Risk ({probability*100:.2f}%)")

        st.info("""
        Recommended actions:
        - Consult a healthcare professional.
        - Monitor blood glucose regularly.
        - Follow dietary recommendations.
        - Maintain regular exercise habits.
        """)

    st.progress(float(probability))
    st.write(f"Estimated probability of diabetes: "f"**{probability*100:.2f}%**")