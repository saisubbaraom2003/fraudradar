from fastapi import FastAPI
from api.schemas import TransactionInput
from api.model_loader import predict_fraud

app = FastAPI(title="FraudRadar API")

@app.get("/")
def root():
    return {"message": "FraudRadar API is running!"}

@app.post("/predict")
def predict(input: TransactionInput):
    if len(input.features) != 29:
        return {"error": "Expected 29 features: V1-V28 + Amount"}

    prediction, probability = predict_fraud(input.features)
    return {
        "prediction": "Fraud" if prediction == 1 else "Not Fraud",
        "probability": probability
    }
