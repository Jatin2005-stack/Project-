from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():

    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    activity = int(request.form["activity"])
    heart_rate = int(request.form["heart_rate"])

    # Simple rule for health
    if temperature > 39.5 or heart_rate > 100:
        result = "Cow is Sick"
    else:
        result = "Cow is Healthy"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)