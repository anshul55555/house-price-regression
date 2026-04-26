import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="House Price Regression",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Enter house details below to predict the estimated house price using the saved `.pkl` regression model.")

MODEL_PATH = Path("house_price_model.pkl")

@st.cache_resource
def load_model_package(model_path: Path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

try:
    model_package = load_model_package(MODEL_PATH)
except FileNotFoundError:
    st.error("`house_price_model.pkl` was not found. Keep this Streamlit file and the PKL file in the same folder.")
    st.stop()

model = model_package["model"]
feature_names = model_package["feature_names"]

st.subheader("Input House Features")

square_footage = st.number_input("Square Footage", min_value=100, max_value=10000, value=2500, step=50)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010, step=1)
lot_size = st.number_input("Lot Size", min_value=0.01, max_value=10.0, value=0.50, step=0.01)
garage_size = st.number_input("Garage Size", min_value=0, max_value=10, value=2, step=1)
neighborhood_quality = st.slider("Neighborhood Quality", min_value=1, max_value=10, value=7)

input_data = pd.DataFrame([{
    "Square_Footage": square_footage,
    "Num_Bedrooms": num_bedrooms,
    "Num_Bathrooms": num_bathrooms,
    "Year_Built": year_built,
    "Lot_Size": lot_size,
    "Garage_Size": garage_size,
    "Neighborhood_Quality": neighborhood_quality
}])

input_data = input_data[feature_names]

if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ₹{prediction:,.2f}")

with st.expander("Model Details"):
    st.write("Model name:", model_package.get("model_name", "Not available"))
    st.write("Feature names:", feature_names)
    if "metrics_on_test_set" in model_package:
        st.dataframe(pd.DataFrame(model_package["metrics_on_test_set"]))
