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
