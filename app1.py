from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and preprocessing objects
model = joblib.load("diabetes_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")


# Define input data structure
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.post("/predict")
def predict(data: DiabetesInput):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "Age"
    ]

    # Replace invalid zero values
    input_df[cols] = input_df[cols].replace(0, np.nan)

    # Apply preprocessing
    input_df[cols] = imputer.transform(input_df[cols])
    input_scaled = scaler.transform(input_df)

    # Get prediction probability
    probability = float(model.predict_proba(input_scaled)[0][1])

    return {
        "probability": probability,
        "prediction": "High Risk" if probability >= 0.30 else "Low Risk"
    }