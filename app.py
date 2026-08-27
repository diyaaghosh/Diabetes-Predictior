from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("diabetes_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([data])

    cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "Age"
    ]

    input_df[cols] = input_df[cols].replace(0, np.nan)
    input_df[cols] = imputer.transform(input_df[cols])
    input_scaled = scaler.transform(input_df)

    probability = float(model.predict_proba(input_scaled)[0][1])

    return jsonify({
        "probability": probability,
        "prediction": "High Risk" if probability >= 0.30 else "Low Risk"
    })

if __name__ == "__main__":
    app.run(debug=True)