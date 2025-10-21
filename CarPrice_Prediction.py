# CarPrice_Prediction.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = Path("best_car_price_model.pkl")
model = joblib.load(MODEL_PATH)

# Feature list the model was trained with (required to align columns)
try:
    FEATURE_NAMES = list(model.feature_names_in_)
except Exception:
    FEATURE_NAMES = None  # Fallback if not available (rare)

# -----------------------------
# Inverse-transform constants
# Standardized cube-root(selling_price)  ->  cube-root  ->  original price
# -----------------------------
MU = 79.56940902399067
SIGMA = 12.463794031640061

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to estimate its selling price (number only, no currency).")

# Categorical choices
brand = st.selectbox(
    "Select Car Brand",
    [
        "Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Ford",
        "BMW", "Audi", "Mahindra", "Datsun", "Renault", "Volkswagen",
        "Chevrolet", "Skoda", "Nissan", "Mercedes-Benz", "Jaguar",
        "Land Rover", "Mini", "Mitsubishi", "Volvo", "Fiat", "Jeep",
        "Porsche", "Ambassador", "Isuzu", "Lexus"
    ],
)

vehicle_age = st.slider("Vehicle Age (years)", 0, 15, 3, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=1000, max_value=300000, value=50000, step=1000)
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=35.0, value=18.0, step=0.1)
engine = st.number_input("Engine Capacity (CC)", min_value=800, max_value=5000, value=1200, step=50)
max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=250.0, value=90.0, step=1.0)
seats = st.selectbox("Number of Seats", [2, 4, 5, 7])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])

# -----------------------------
# Building a model input row which will align to training features
# -----------------------------
def build_feature_row():
    """
    Create a single-row DataFrame with columns exactly matching model.feature_names_in_.
    Unknown OHE columns are kept at 0; selected categories are set to 1.
    Numeric columns are set directly from inputs.
    """
    if FEATURE_NAMES is None:
        base_cols = [
            "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats",
            f"fuel_type_{fuel_type}", f"transmission_type_{transmission_type}",
            f"brand_{brand}",
        ]
        data = {col: 0 for col in base_cols}
        data["vehicle_age"] = vehicle_age
        data["km_driven"] = km_driven
        data["mileage"] = mileage
        data["engine"] = engine
        data["max_power"] = max_power
        data["seats"] = seats
        data[f"fuel_type_{fuel_type}"] = 1
        data[f"transmission_type_{transmission_type}"] = 1
        data[f"brand_{brand}"] = 1
        return pd.DataFrame([data])

    # If we have the exact feature list, we can create a zero vector and fill appropriately
    row = pd.Series(0.0, index=FEATURE_NAMES, dtype=float)

    # Set numeric features if they exist in the model's feature set
    for name, val in {
        "vehicle_age": float(vehicle_age),
        "km_driven": float(km_driven),
        "mileage": float(mileage),
        "engine": float(engine),
        "max_power": float(max_power),
        "seats": float(seats),
    }.items():
        if name in row.index:
            row[name] = val

    # One-hot slots — set to 1 only if the column exists
    cat_map = {
        f"fuel_type_{fuel_type}": 1.0,
        f"transmission_type_{transmission_type}": 1.0,
        f"brand_{brand}": 1.0,
    }
    for col, v in cat_map.items():
        if col in row.index:
            row[col] = v

    return pd.DataFrame([row.values], columns=row.index)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Price"):
    X = build_feature_row()

    # Model outputs standardized cube-root(price): y_std
    y_std = float(model.predict(X)[0])

    # Invert: standardized -> cube-root -> original price
    y_cuberoot = y_std * SIGMA + MU
    price = y_cuberoot ** 3

    st.success(f"Estimated Selling Price: {price:,.2f}")
