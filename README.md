#  Diabetes Prediction App



**AI-powered early health risk detection system** built using **Streamlit**, **Python**, and **Machine Learning**.  
This app predicts the likelihood of diabetes based on user health inputs such as glucose level, insulin, BMI, and age.

---

##  Features

-  Clean and interactive **Streamlit UI**
-  Accepts multiple health parameters
-  Uses the **best-trained ML model** (XGBoost)
-  Displays confidence score and prediction message and Advice



---

##  Model Information

The backend model was trained on the **Pima Indians Diabetes Dataset**, using multiple algorithms:
- Logistic Regression  
- Random Forest  
- XGBoost  


After evaluation, the model with **highest AUC and F1-score** was chosen as the best and saved as:


---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **Model Saving** | joblib |
| **Dataset** | Pima Indians Diabetes Dataset (UCI Repository) |

---

##  Input Parameters

| Parameter | Description |
|------------|-------------|
| Pregnancies | Number of times pregnant (if applicable) |
| Glucose Level | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin Level | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| Diabetes Pedigree Function | Genetic likelihood of diabetes |
| Age | Age in years |

---

##  Installation

### 1️ Clone the Repository
```bash
git clone https://github.com/diyaaghosh/diabetes-predictor.git
```

### **Install Dependencies**
```
pip install -r requirements.txt
```
### **Run the Streamlit App**
```
streamlit run app.py
```
### **File Structure**
```

├── chat.py                      # Streamlit app main file
├── model_columns.pkl
├── diabetes_model.pkl   # Saved ML model
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── dataset.csv                 # (optional) Training dataset
|__ scaer.pkl
|__ imputer.pkl
|
```


### **Dataset**
```
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
```