# app.py (REVISED for 4 features)
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import sys

print("--- Flask app startup sequence initiated ---")

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('parkinsons_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("SUCCESS: Model and scaler loaded successfully.")
except FileNotFoundError:
    print("ERROR: One or both of 'parkinsons_rf_model.pkl' or 'scaler.pkl' not found.")
    print("Please ensure you have run 'python train_model.py' to create these files in the same directory as app.py.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading model or scaler: {e}")
    sys.exit(1)

# --- IMPORTANT: Define ONLY the 4 expected feature names here ---
feature_names = ['PPE', 'spread1', 'spread2', 'MDVP:Fo(Hz)'] # <--- YOUR CHOSEN 4 FEATURES

print(f"INFO: Expected feature names configured: {len(feature_names)} features: {feature_names}")


@app.route('/')
def home():
    print("INFO: '/' route accessed. Rendering index.html.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("INFO: '/predict' route accessed via POST.")
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            input_values = []
            
            print(f"DEBUG: Received form data: {data}")

            # Ensure all expected features are present and valid
            for feature in feature_names: # Loop through only the 4 expected features
                value = data.get(feature)
                if value is None or value == '':
                    print(f"WARNING: Missing or empty value for feature: {feature}")
                    return render_template('result.html', prediction_text=f"Error: Missing value for {feature}", confidence_score="N/A")
                try:
                    input_values.append(float(value))
                except ValueError:
                    print(f"WARNING: Invalid number format for feature: {feature} (value: '{value}')")
                    return render_template('result.html', prediction_text=f"Error: Invalid number for {feature}", confidence_score="N/A")

            # Convert to numpy array and reshape for single prediction
            input_array = np.array(input_values).reshape(1, -1)
            print(f"DEBUG: Input array created: {input_array.shape}")

            # Scale the input features using the loaded scaler
            scaled_input = scaler.transform(input_array) # Scaler expects only these 4 features now
            print(f"DEBUG: Input scaled: {scaled_input.shape}")

            # Make prediction
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]
            
            print(f"DEBUG: Raw prediction: {prediction}, Probabilities: {prediction_proba}")

            result = "Parkinson's Detected" if prediction == 1 else "No Parkinson's Detected"
            confidence = f"{prediction_proba[prediction]*100:.2f}%"
            
            print(f"INFO: Prediction result: {result}, Confidence: {confidence}")
            return render_template('result.html', prediction_text=result, confidence_score=confidence)

        except Exception as e:
            print(f"CRITICAL ERROR: An unhandled exception occurred in the /predict route: {e}")
            import traceback
            traceback.print_exc()
            return render_template('result.html', prediction_text=f"An unexpected server error occurred: {e}", confidence_score="N/A")

if __name__ == '__main__':
    print("--- Attempting to run Flask development server ---")
    try:
        app.run(debug=True, port=7000, use_reloader=False)
        print("INFO: Flask app.run() has returned. This should not happen if the server started successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Flask server failed to start: {e}")
        import traceback
        traceback.print_exc()
    print("--- Flask app startup sequence finished ---")