import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Car Price Prediction")

wheelbase = st.number_input("Wheelbase", min_value=50.0)
horsepower = st.number_input("Horsepower", min_value=1)

fuel = st.selectbox("Fuel Type", ["Gas", "Diesel"])

if fuel == "Gas":
    fueltype_gas = 1
    fueltype_diesel = 0
else:
    fueltype_gas = 0
    fueltype_diesel = 1

if st.button("Predict Price"):
    features = np.array([[
        wheelbase,
        horsepower,
        fueltype_diesel,
        fueltype_gas
    ]])

    prediction = model.predict(features)

    st.success(f"Predicted Car Price: ₹{prediction[0]:,.0f}")