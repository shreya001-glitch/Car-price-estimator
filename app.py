import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

print(np.__version__)
print(sklearn.__version__)

# Load model
model = pickle.load(open("car_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🚗 Car Price Prediction")

st.write("Enter car details below")

# Inputs

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2025,
    value=2018
)

km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000
)

fuel = st.selectbox(
    "Fuel Type",
    ["Diesel", "Petrol", "CNG", "LPG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Individual", "Dealer", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

mileage = st.number_input(
    "Mileage (km/l)",
    min_value=0.0,
    value=20.0
)

engine = st.number_input(
    "Engine CC",
    min_value=500,
    value=1200
)

max_power = st.number_input(
    "Max Power",
    min_value=10.0,
    value=80.0
)

seats = st.number_input(
    "Seats",
    min_value=2,
    max_value=10,
    value=5
)

brand = st.text_input(
    "Brand",
    value="Maruti"
)

# Prediction Button

if st.button("Predict Price"):

    data = {
        "year": year,
        "km_driven": km_driven,
        "mileage(km/ltr/kg)": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner
    }

    input_df = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]

    st.success(
        f"Estimated Selling Price: ₹{prediction:,.0f}"
    )