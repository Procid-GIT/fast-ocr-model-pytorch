import os
import sys
from flask import Flask, jsonify, request, app
from flask_cors import CORS
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from model_prediction_wrapper import predict

app = Flask(__name__)

CORS(app, origins=["https://Procid-GIT.github.io/OCR-Digits-Model"])

@app.route('/api/predict', methods=['POST'])
def get_ai_prediction():
    image = request.files.get('user_file')
    if not image:
        return jsonify({"prediction": "Error"}), 400
    ai_prediction = predict(image.stream)
    return jsonify({"prediction": ai_prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)