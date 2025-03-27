import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import re
import plotly.graph_objects as go

# Streamlit Page Config
st.set_page_config(page_title="Stock Price Prediction", layout="centered")

# Title
st.title("📈 Stock Price Prediction App")
st.markdown("Predict the next 7 days of stock prices based on historical data using Linear Regression.")

# Utility: Normalize inconsistent date formats to mm-dd-yyyy
def normalize_date(date_str):
    match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return f"{int(month):02d}-{int(day):02d}-{year}"
    return date_str

# Load and clean stock data
df = pd.read_csv("stock_data.csv")
df['Date'] = df['Date'].astype(str).apply(normalize_date)
df['Date'] = pd.to_datetime(df['Date'], format="%m-%d-%Y", errors='coerce')

# Check for date parsing errors
if df['Date'].isnull().any():
    st.error("❌ Some dates in the data couldn't be parsed. Please fix your CSV and try again.")
    st.stop()

# Convert to ordinal
df['DateOrdinal'] = df['Date'].map(lambda d: d.toordinal())

# Get available stocks
available_stocks = df['Stock'].unique()

# Select stocks
selected_stocks = st.multiselect("📊 Select stocks to predict:", available_stocks)

# Forecast horizon
forecast_days = 7

# Create result dataframe and chart
forecast_result = []
fig = go.Figure()

# Predict for each selected stock
for stock in selected_stocks:
    model_path = f"models/{stock}_model.pkl"
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file not found for stock '{stock}'. Please check if it's trained.")
        continue

    # Load model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Subset data for the stock
    stock_df = df[df['Stock'] == stock].sort_values(by='Date')
    
    # Determine the last available date
    last_date = stock_df['Date'].max()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    forecast_ordinals = np.array([[date.toordinal()] for date in forecast_dates])
    
    # Predict for 7 days
    predictions = model.predict(forecast_ordinals)

    # Append to results
    for date, price in zip(forecast_dates, predictions):
        forecast_result.append({
            "Stock": stock,
            "Date": date.strftime("%d-%m-%Y"),
            "Predicted Price (₹)": round(price, 2)
        })

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=stock_df['Date'],
        y=stock_df['Price'],
        mode='lines+markers',
        name=f"{stock} - Historical",
        hovertemplate="Stock: " + stock + "<br>Date: %{x|%d-%m-%Y}<br>Price: ₹%{y:.2f}<extra></extra>"
    ))

    # Plot forecast data
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode='lines+markers',
        name=f"{stock} - Forecast",
        line=dict(dash='dash'),
        marker=dict(symbol='x'),
        hovertemplate="Stock: " + stock + "<br>Date: %{x|%d-%m-%Y}<br>Forecast: ₹%{y:.2f}<extra></extra>"
    ))

# Show chart
if selected_stocks:
    fig.update_layout(
        title="📊 Stock Price Trend + 1-Week Forecast",
        xaxis_title="Date",
        yaxis_title="Stock Price (₹)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show forecast table
    forecast_df = pd.DataFrame(forecast_result)
    st.subheader("📋 1-Week Forecast Table")
    st.dataframe(forecast_df)
else:
    st.info("👈 Please select at least one stock to view predictions.")
