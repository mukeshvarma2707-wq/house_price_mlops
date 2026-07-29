import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data[["Area"]]
y = data["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "house_price_model.pk")



