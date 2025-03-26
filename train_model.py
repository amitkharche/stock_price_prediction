import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv("stock_data.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['DateOrdinal'] = df['Date'].map(lambda date: date.toordinal())

X = df[['DateOrdinal']]
y = df['Price']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("stock_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'stock_price_model.pkl'")
