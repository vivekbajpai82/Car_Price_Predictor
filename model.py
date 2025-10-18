import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# Use relative path for CSV file
data_path = os.path.join('Cleaned_Car_data.csv')
df = pd.read_csv(data_path)
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Encode categorical variables
label_encoders = {}
for col in ['name', 'company', 'fuel_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=['Price'])
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save model and encoders
os.makedirs('model', exist_ok=True)
joblib.dump(model, os.path.join('model', 'car_price_model.pkl'))
joblib.dump(label_encoders, os.path.join('model', 'encoders.pkl'))

print("✅ Model and encoders saved successfully in 'model/' folder.")
