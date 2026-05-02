import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

# Get exact feature names from trained model
feature_names = model.named_steps['scaler'].feature_names_in_

st.title("📊 Customer Churn Predictor")
st.write("Fill in customer details to predict if they will churn.")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
contract = st.selectbox("Contract Type", [0, 1, 2], 
    format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
internet = st.selectbox("Internet Service", [0, 1, 2], 
    format_func=lambda x: ["DSL", "Fiber optic", "No"][x])
payment = st.selectbox("Payment Method", [0, 1, 2, 3])

if st.button("Predict Churn"):
    # Create a row of zeros with correct feature names
    sample = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    
    # Fill in the values we have
    if "tenure" in feature_names:
        sample["tenure"] = tenure
    if "MonthlyCharges" in feature_names:
        sample["MonthlyCharges"] = monthly
    if "TotalCharges" in feature_names:
        sample["TotalCharges"] = total
    if "Contract" in feature_names:
        sample["Contract"] = contract
    if "InternetService" in feature_names:
        sample["InternetService"] = internet
    if "PaymentMethod" in feature_names:
        sample["PaymentMethod"] = payment

    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to CHURN ({round(proba*100)}% probability)")
    else:
        st.success(f"✅ This customer is likely to STAY ({round((1-proba)*100)}% probability)")