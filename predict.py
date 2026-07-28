import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Example prediction
area = 2300

prediction = model.predict([[area]])

print(f"Area : {area} sq.ft")
print(f"Predicted Price : ${prediction[0]:,.2f}")