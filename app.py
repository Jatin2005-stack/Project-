from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()

@app.route('/')
def home():
    return render_template("index.html", accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    hum = float(request.form['humidity'])
    act = float(request.form['activity'])
    hr = float(request.form['heart_rate'])

    prediction = model.predict([[temp, hum, act, hr]])

    if prediction[0] == 1:
        result = "⚠️ ALERT: Cow is SICK"
    else:
        result = "✅ Cow is HEALTHY"

    return render_template("index.html", prediction_text=result, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)True)
