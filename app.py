import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set the page config
st.set_page_config(page_title="Stock Price Prediction", layout="centered")

# Load trained model
with open("stock_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load historical stock data
df = pd.read_csv("stock_data.csv")

# Handle date format in dd-mm-yyyy
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df['DateOrdinal'] = df['Date'].map(lambda date: date.toordinal())

# App title
st.title("📈 Stock Price Prediction App")
st.markdown("Predict future stock price based on historical data using Linear Regression.")

# Get date range
min_date = df['Date'].min()
max_date = df['Date'].max() + timedelta(days=30)

# Date input from user
selected_date = st.date_input(
    "📅 Select a future date to predict stock price:",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

# Convert selected date to ordinal
selected_date_ordinal = np.array([[selected_date.toordinal()]])

# Predict using the model
predicted_price = model.predict(selected_date_ordinal)[0]

# Display prediction
st.subheader(f"💰 Predicted Price on {selected_date.strftime('%d-%m-%Y')}: ₹{predicted_price:.2f}")

# Plotting the historical prices + prediction
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Date'], df['Price'], label="📊 Historical Prices", marker='o')
ax.scatter(selected_date, predicted_price, color='red', label="🔮 Predicted Price", s=100, zorder=5)
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price (₹)")
ax.set_title("Stock Price Trend")
ax.legend()
ax.grid(True)

# Show plot
st.pyplot(fig)
