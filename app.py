from flask import Flask, request, jsonify
import numpy as np
import joblib
import traceback

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Prediction API. Use the /predict endpoint with a POST request to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check content type
        if request.content_type != 'application/json':
            return jsonify({'error': 'Content-Type must be application/json'}), 415

        # Parse JSON data
        data = request.json
        if 'features' not in data:
            return jsonify({'error': "'features' key missing in request body"}), 400

        features = np.array(data['features']).reshape(1, -1)

        # Load model and scaler
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Scale features and predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Log stack trace for debugging
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
