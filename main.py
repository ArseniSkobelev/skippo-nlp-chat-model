import glob
import json
import os

import numpy as np
from flask import Flask, jsonify, request, Response
import joblib

app = Flask(__name__)

model = None

_threshold = .4


def load_model():
    model_files = glob.glob('models/model-*.joblib')
    if not model_files:
        raise FileNotFoundError("No model files found.")

    sorted_model_files = sorted(model_files, key=os.path.getmtime)

    latest_model_file = sorted_model_files[-1]

    global model
    model = joblib.load(latest_model_file)


def predict_with_fallback(model_repr, input_data_param, threshold_param, fallback_value):
    proba = model_repr.predict_proba(input_data_param)

    max_proba = np.max(proba, axis=1)

    if np.any(max_proba > threshold_param):
        predictions = model_repr.predict(input_data_param)
        return predictions[0]

    else:
        return fallback_value


@app.route("/ping", methods=["GET"])
def test():
    return jsonify({"message": "Pong!"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    user_input = data['user_input']

    if not user_input:
        json_data = {
            "success": False,
            "message": ""
        }

        return Response(
            json.dumps(json_data),
            status=500,
            mimetype="application/json"
        )

    prediction = predict_with_fallback(model, [user_input], _threshold, "General.Output.UnknownCommand")

    json_data = {
        "success": True,
        "message": prediction
    }

    return Response(
        json.dumps(json_data),
        status=200,
        mimetype="application/json"
    )


if __name__ == '__main__':
    load_model()
    app.run(debug=False)
