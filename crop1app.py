import streamlit as st
import pickle
import numpy as np

# Load the trained RandomForest model and scaler
with open('crop.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sccrop.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the web app
st.title('🌾 Crop Recommendation App')

st.write("Enter soil and weather conditions to get the best crop recommendation.")

# Input fields
N = st.number_input('Nitrogen (N)', min_value=0.0, value=0.0, step=1.0)
P = st.number_input('Phosphorus (P)', min_value=0.0, value=0.0, step=1.0)
K = st.number_input('Potassium (K)', min_value=0.0, value=30.0, step=1.0)
temperature = st.number_input('Temperature (°C)', min_value=0.0, value=25.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0.0, value=50.0, step=0.1)
ph = st.number_input('pH Level', min_value=0.0, value=6.5, step=0.1)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=100.0, step=1.0)

# Button to trigger prediction
if st.button('Predict Crop'):
    # Prepare the feature vector
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=np.float64)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict Crop
    predicted_crop = model.predict(features_scaled)

    # Display the prediction
    st.success(f'🌱 Recommended Crop: **{predicted_crop[0]}**')
