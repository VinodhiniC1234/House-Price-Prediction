import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction Dashboard")

# ---------------- LOAD DATA ---------------- #
file_path = "dataset.csv"

if not os.path.exists(file_path):
    st.error("❌ dataset.csv not found in project folder")
    st.stop()

df = pd.read_csv(file_path)

if df.empty:
    st.error("❌ dataset.csv is empty")
    st.stop()

st.success("✅ Dataset loaded successfully!")
st.dataframe(df)

# ---------------- SIMPLE MODEL ---------------- #
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- USER INPUT ---------------- #
st.subheader("🔢 Predict House Price")

area = st.number_input("Area (sq ft)", 500, 10000, 1500)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
stories = st.number_input("Stories", 1, 5, 2)
parking = st.number_input("Parking spaces", 0, 5, 1)

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms, stories, parking]])
    st.success(f"💰 Predicted Price: ₹ {int(prediction[0]):,}")