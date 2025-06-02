import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Crop_recommendation (1).csv')

# Split data into features (X) and labels (y)
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # Last column (crop labels)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
with open('crop.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('sccrop.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and scaler saved successfully!")
