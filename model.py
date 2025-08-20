import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\ashut\OneDrive\Desktop\ml2\Cleaned_Car_data.csv")
df = df.drop(columns=['Unnamed: 0'])

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
import joblib

# Save the model to a file
joblib.dump(model, 'car_price_model.pkl')
print("✅ Model saved as car_price_model.pkl")
# Save the label encoders
encoders_to_save = {col: le for col, le in label_encoders.items()}
joblib.dump(encoders_to_save, 'encoders.pkl')
print("✅ Encoders saved as encoders.pkl")

