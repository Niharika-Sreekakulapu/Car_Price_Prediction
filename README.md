# 🚗 Car Price Prediction using Deep Learning

A deep learning-based web app that predicts the price of a used car based on various features like year, fuel type, transmission, ownership, and seller type. Built with Keras and deployed using Flask.

## 💡 Features
- Deep Learning model trained on a real-world dataset (`cardekho.csv`)
- Input features like year, fuel, seller type, owner, transmission
- Flask-based frontend to interact with the model
- Encoders and scalers stored as `.save` files for real-time inference

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Flask
- HTML/CSS (Jinja2 templates)

## 📁 Project Structure
ML/
├── pycache/
│ ├── app.cpython-311.pyc
│ └── cars.cpython-311.pyc
├── model/
│ └── car_price_model.h5
├── templates/
│ ├── index.html
│ └── result.html
├── .gitignore
├── accuracy.txt
├── app.py
├── cardekho.csv
├── cars.py
├── le_fuel.save
├── le_owner.save
├── le_seller_type.save
├── le_seller.save
├── le_trans.save
├── le_transmission.save
├── output.docx
├── README.md
├── requirements.txt
├── scaler.save
├── scaler_X.save
└── scaler_y.save

📸 Demo
![Car Price Prediction UI](input_form.png)
![Car Price Prediction UI](output.png.png)

👩‍💻 Author
Niharika Sreekakulapu