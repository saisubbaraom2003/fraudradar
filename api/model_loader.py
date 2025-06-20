import joblib
import os

MODEL_PATH = os.path.join("models", "fraud_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_fraud(features):
    # Features already scaled except 'Amount'
    scaled_amount = scaler.transform([[features[-1]]])[0][0]
    features[-1] = scaled_amount

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    return int(prediction), round(probability, 4)
