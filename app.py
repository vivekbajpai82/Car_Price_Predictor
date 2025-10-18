from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define model path safely for Render/Linux
model_path = os.path.join('model', 'car_price_model.pkl')
encoder_path = os.path.join('model', 'encoders.pkl')

# Load model and encoders
model = joblib.load(model_path)
encoders = joblib.load(encoder_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    company = request.form['company']
    year = int(request.form['year'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']

    # Encode categorical variables
    try:
        name_encoded = encoders['name'].transform([name])[0]
        company_encoded = encoders['company'].transform([company])[0]
        fuel_encoded = encoders['fuel_type'].transform([fuel_type])[0]
    except:
        return render_template('index.html', prediction_text="❌ Invalid input. Please use known values.")

    # Model prediction
    input_data = np.array([[name_encoded, company_encoded, year, kms_driven, fuel_encoded]])
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f"💰 Estimated Price: ₹{int(prediction):,}")


if __name__ == "__main__":
    # Don't use debug=True on Render
    app.run(host='0.0.0.0', port=10000)
