import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data/bovine_health_dataset_2000_rows.csv")

# Features and target
X = data[['temperature', 'humidity', 'activity', 'heart_rate']]
y = data['disease']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route (IMPORTANT CHANGE: JSON return)
@app.route('/check', methods=["POST"])
def check():
    try:
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        activity = int(request.form.get("activity"))
        heart_rate = int(request.form.get("heart_rate"))

        input_data = [[temperature, humidity, activity, heart_rate]]
        prediction = model.predict(input_data)

        result = prediction[0]

        return jsonify({"result": result})   # ✅ JSON return

    except Exception as e:
        return jsonify({"result": "Error: " + str(e)})

# Run app
if __name__ == "__main__":
    app.run(debug=True)
