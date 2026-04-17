from flask import Flask, request, jsonify
import joblib

application = Flask(__name__)

# Load model
model = joblib.load("sentiment_model.joblib")

@application.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    prediction = model.predict([text])[0]

    return jsonify({
        "input_text": text,
        "prediction": str(prediction)
    })

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)