# app.py
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

model = load_model('car_price_model.h5')
scaler_X = joblib.load('scaler_X.save')
scaler_y = joblib.load('scaler_y.save')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        km_driven = float(request.form['km_driven'])
        fuel = int(request.form['fuel'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])
        mileage = float(request.form['mileage'])
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = int(request.form['seats'])

        input_data = np.array([[year, km_driven, fuel, seller_type, transmission, owner,
                                mileage, engine, max_power, seats]])
        input_scaled = scaler_X.transform(input_data)
        pred_scaled = model.predict(input_scaled)
        predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]

        with open("accuracy.txt", "r") as f:
            accuracy = f.read()

        return render_template('result.html', price=round(predicted_price, 2), accuracy=accuracy)
    
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
