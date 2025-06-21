import streamlit as st
import requests

st.set_page_config(page_title="FraudRadar", page_icon="💳")
st.title("💳 FraudRadar: Real-Time Fraud Detection")
st.markdown("Enter transaction details below to detect potential fraud:")

# Feature names (29 features expected by your backend model)
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

# Input fields
features = [st.number_input(name, value=0.0) for name in feature_names]

# Use your actual deployed backend here 👇
API_URL = "https://fraudradar.onrender.com/predict"

# Submit to FastAPI
if st.button("Predict"):
    with st.spinner("Sending data to FraudRadar backend..."):
        try:
            res = requests.post(API_URL, json={"features": features})
            if res.status_code == 200:
                result = res.json()
                st.success(f"✅ Prediction: {result['prediction']}")
                st.info(f"🔍 Fraud Probability: {result['probability']:.2%}")
            else:
                st.error(f"❌ API Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"🚫 Connection Error: {e}")
