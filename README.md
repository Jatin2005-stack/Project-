# Project-import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset/bovine_health_dataset.csv")

# Features and target
X = data.drop("health_status", axis=1)
y = data["health_status"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/bovine_health_model.pkl")

print("Model trained and saved successfully!")
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/bovine_health_model.pkl")

def predict_disease(temperature, humidity, activity, heart_rate):

    data = np.array([[temperature, humidity, activity, heart_rate]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        return "Cow is Sick"
    else:
        return "Cow is Healthy"


# Example test
result = predict_disease(39.5, 70, 30, 90)

print(result)

