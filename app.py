from flask import Flask, render_template, request, jsonify
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from twilio.rest import Client

app = Flask(__name__)

# TWILIO
ACCOUNT_SID = "AC816cdeac76ad2044b72c7763dccbe100"
AUTH_TOKEN = "8bdacdfa9b7e3c29496c25f0e7503708"
TWILIO_NUMBER = "+12605253378"
MY_NUMBER = "+919350847738"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Load model
model = joblib.load("model.pkl")
kmeans = joblib.load("kmeans.pkl")

# Accuracy
try:
    with open("accuracy.txt") as f:
        accuracy = f.read()
except:
    accuracy = "No data"

last_data = {}

@app.route('/')
def home():
    return render_template("index.html",
                           accuracy=accuracy,
                           acc_graph="accuracy_graph.png",
                           cm="confusion_matrix.png")

@app.route('/predict', methods=['POST'])
def predict():
    global last_data

    temp = float(request.form['temperature'])
    hum = float(request.form['humidity'])
    act = float(request.form['activity'])
    hr = float(request.form['heart_rate'])

    last_data = {"temp": temp, "hum": hum, "act": act, "hr": hr}

    df = pd.DataFrame([[temp, hum, act, hr]],
                      columns=['temperature','humidity','activity','heart_rate'])

    # 🔥 PURE ML
    pred = model.predict(df)[0]
    cluster = kmeans.predict(df)[0]

    if pred == 1:
        result = "⚠️ Cow is SICK"
        color = "red"
    else:
        result = "✅ Cow is HEALTHY"
        color = "green"

    # SMS
    try:
        if pred == 1:
            client.messages.create(
                body=f"Cow Sick!\nTemp={temp}, HR={hr}",
                from_=TWILIO_NUMBER,
                to=MY_NUMBER
            )
        alert = "Alert processed"
    except:
        alert = "SMS error"

    # Graph
    plt.figure()
    plt.bar(['Temp','Humidity','Activity','HR'], [temp, hum, act, hr], color=color)
    plt.savefig("static/live_graph.png")
    plt.close()

    return render_template("index.html",
                           prediction_text=result,
                           alert=alert,
                           accuracy=accuracy,
                           graph="live_graph.png",
                           acc_graph="accuracy_graph.png",
                           cm="confusion_matrix.png",
                           cluster=f"Cluster: {cluster}")

@app.route('/chat', methods=['POST'])
def chat():
    global last_data

    msg = request.json.get("message", "").lower()

    if not last_data:
        return jsonify({"reply": "First predict data."})

    temp = last_data["temp"]
    hr = last_data["hr"]

    if "sick" in msg:
        reply = f"Temp={temp}, HR={hr}. Possible illness detected."
    else:
        reply = "Ask about cow health."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
