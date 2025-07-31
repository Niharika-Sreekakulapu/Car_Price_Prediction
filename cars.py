# cars.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
data = pd.read_csv('cardekho.csv')
data = data.replace(r'^\s*$', np.nan, regex=True)

for col in ['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

# Encode categorical features
encoders = {
    'fuel': LabelEncoder(),
    'seller_type': LabelEncoder(),
    'transmission': LabelEncoder(),
    'owner': LabelEncoder()
}
for col, encoder in encoders.items():
    data[col] = encoder.fit_transform(data[col])
    joblib.dump(encoder, f'le_{col}.save')

X = data[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner',
          'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']]
y = data['selling_price'].values.reshape(-1, 1)

# Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('car_price_model.h5')
model.summary()
# Model accuracy
y_pred_scaled = model.predict(X_test)
r2 = r2_score(y_test, y_pred_scaled)
print(f"✅ Model Accuracy: {r2 * 100:.2f}%")

# Save accuracy to a text file for Flask to read
with open('accuracy.txt', 'w') as f:
    f.write(f"{r2 * 100:.2f}")
