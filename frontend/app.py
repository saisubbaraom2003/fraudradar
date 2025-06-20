import streamlit as st
import requests

st.title("💳 FraudRadar: Real-Time Fraud Detection")

st.markdown("Enter transaction features:")

feature_names = [
    "Transaction Time",
    "Transaction Amount",
    "Customer Age",
    "Merchant Category",
    "Card Present",
    "Transaction Type",
    "Transaction Channel",
    "Customer Location Distance",
    "Merchant Location",
    "Cardholder Account Age",
    "Number of Prior Transactions",
    "Average Transaction Amount",
    "Max Transaction Amount",
    "Transaction Hour of Day",
    "Transaction Day of Week",
    "Merchant Risk Score",
    "Cardholder Risk Score",
    "Device Type",
    "IP Address Risk",
    "Velocity of Transactions",
    "Previous Fraud Flags",
    "Transaction Currency",
    "Merchant Category Code",
    "Payment Method",
    "Authorization Status",
    "Device Location",
    "Card Type",
    "Account Balance",
    "Transaction Amount (in USD)"
]

features = []

for name in feature_names:
    val = st.number_input(name, value=0.0)
    features.append(val)

if st.button("Predict"):
    with st.spinner("Predicting..."):
        payload = {"features": features}
        try:
            res = requests.post("http://localhost:8000/predict", json=payload)
            result = res.json()
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Fraud Probability: {result['probability']}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
