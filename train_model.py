import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

# Step 1: Normalize inconsistent date formats
def normalize_date(date_str):
    match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return f"{int(month):02d}-{int(day):02d}-{year}"
    return date_str  # Return original if not matching

# Load raw data
df = pd.read_csv("stock_data.csv")

# Normalize and parse dates
df['Date'] = df['Date'].astype(str).apply(normalize_date)
df['Date'] = pd.to_datetime(df['Date'], format="%m-%d-%Y", errors='coerce')

# Check for unparseable dates
if df['Date'].isnull().any():
    print("❌ Some dates still couldn't be parsed:")
    print(df[df['Date'].isnull()])
    exit("🚫 Please fix the above date issues in your CSV before training.")

# Step 2: Fill missing dates for each stock with previous day's price
all_dates = pd.date_range(df['Date'].min(), df['Date'].max())
stocks = df['Stock'].unique()
filled_df_list = []

for stock in stocks:
    stock_df = df[df['Stock'] == stock].copy()
    stock_df.set_index('Date', inplace=True)
    stock_df = stock_df[['Price']].reindex(all_dates)
    stock_df['Stock'] = stock
    stock_df['Price'] = stock_df['Price'].ffill()  # Forward fill missing prices
    stock_df.reset_index(inplace=True)
    stock_df.rename(columns={'index': 'Date'}, inplace=True)
    filled_df_list.append(stock_df)

# Combine all filled stock data
df_filled = pd.concat(filled_df_list, ignore_index=True)

# Convert date to ordinal for regression
df_filled['DateOrdinal'] = df_filled['Date'].map(lambda date: date.toordinal())

# Create directory to save models
os.makedirs("models", exist_ok=True)

# Step 3: Train and save model per stock
for stock in stocks:
    stock_df = df_filled[df_filled['Stock'] == stock]
    X = stock_df[['DateOrdinal']]
    y = stock_df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    model_filename = f"models/{stock}_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved for stock: {stock} → {model_filename}")

print("🎉 All stock models trained and saved successfully.")
