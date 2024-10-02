# pip install fastapi uvicorn joblib
# uvicorn main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load('fraud_detection_model_v1.pkl')
scaler = joblib.load('scaler_v1.pkl')

# Initialize FastAPI
app = FastAPI()

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post('/predict')
def predict(transaction: Transaction):
    # Convert the transaction data to a DataFrame
    data = pd.DataFrame([transaction.dict()])
    
    # Separate 'Time' and 'Amount' for scaling
    time_amount = data[['Time', 'Amount']]
    other_features = data.drop(['Time', 'Amount'], axis=1)
    
    # Scale 'Time' and 'Amount' using the saved scaler
    time_amount_scaled = scaler.transform(time_amount)
    time_amount_scaled_df = pd.DataFrame(time_amount_scaled, columns=['Time', 'Amount'])
    
    # Combine scaled and other features
    processed_data = pd.concat([time_amount_scaled_df, other_features], axis=1)
    
    # Ensure the feature order matches the training data
    feature_order = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    processed_data = processed_data[feature_order]
    
    # Make prediction
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)[:, 1]
    
    # Return the result
    result = {
        'prediction': int(prediction[0]),
        'probability': float(prediction_proba[0])
    }
    return result