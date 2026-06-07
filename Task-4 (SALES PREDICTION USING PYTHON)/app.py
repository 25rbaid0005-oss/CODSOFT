import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("sales_model.pkl")

st.set_page_config(page_title="Sales Prediction", page_icon="📈")

st.title("📈 Sales Prediction using Machine Learning")

st.write("""
Predict sales based on advertising expenditure on:

- TV
- Radio
- Newspaper
""")

tv = st.number_input("TV Advertisement Budget", min_value=0.0, value=100.0)
radio = st.number_input("Radio Advertisement Budget", min_value=0.0, value=20.0)
newspaper = st.number_input("Newspaper Advertisement Budget", min_value=0.0, value=10.0)

if st.button("Predict Sales"):
    data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(data)[0]

    st.success(f"Predicted Sales: {prediction:.2f}")