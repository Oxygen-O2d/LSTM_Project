import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import os

# --- Configuration ---
MODEL_FILE = 'lstm_aqi_model.h5'
SCALER_FILE = 'scaler.pkl'
LOOK_BACK = 24  # Must be the same as used in training
# The number of features the model was trained on (pm2.5, DEWP, TEMP, etc. + one-hot encoded wind)
# pm2.5, DEWP, TEMP, PRES, Iws, Is, Ir, wind_cv, wind_NE, wind_NW, wind_SE -> 11 features
EXPECTED_FEATURES = 11

# --- Helper Functions ---

def get_aqi_category(pm25):
    """Returns the AQI category and color for a given PM2.5 value."""
    if 0 <= pm25 <= 50:
        return "Good", "#00e400"  # Green
    elif 51 <= pm25 <= 100:
        return "Moderate", "#ffff00"  # Yellow
    elif 101 <= pm25 <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"  # Orange
    elif 151 <= pm25 <= 200:
        return "Unhealthy", "#ff0000"  # Red
    elif 201 <= pm25 <= 300:
        return "Very Unhealthy", "#8f3f97"  # Purple
    else:
        return "Hazardous", "#7e0023"  # Maroon

# --- Load Model and Scaler ---
@st.cache_resource
def load_prediction_model():
    """Loads the trained LSTM model and scaler."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return None, None, None
        
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        # Create a sample historical sequence for the app to use
        # This simulates the 23 hours of data leading up to the user's input
        sample_history = np.random.rand(LOOK_BACK, EXPECTED_FEATURES)
        return model, scaler, sample_history
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="AQI Forecaster", layout="wide", initial_sidebar_state="collapsed")

# Load model, scaler, and sample history
model, scaler, sample_history = load_prediction_model()

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(to right, #6dd5ed, #2193b0);
        color: white;
    }
    .stApp {
         background: linear-gradient(to right, #6dd5ed, #2193b0);
    }
    .stMetric, .stNumberInput, .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        color: white !important;
    }
    .stButton>button {
        background-image: linear-gradient(to right, #ff416c 0%, #ff4b2b 51%, #ff416c 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 15px 0;
        width: 100%;
        transition: 0.5s;
        background-size: 200% auto;
    }
    .stButton>button:hover {
        background-position: right center;
    }
    .container {
        padding: 2rem;
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    h1, h2, h3, .stMarkdown, .stSelectbox>div>div {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
if model is not None:
    st.title("🌬️ Air Quality (AQI) Forecaster")
    st.markdown("Predict the PM2.5 concentration for the next hour using a sophisticated LSTM model. Enter the current weather and pollution data to get a forecast.")
    st.markdown("---")

    # --- User Input Section ---
    with st.container():
        st.header("📍 Input Current Conditions")
        
        cols = st.columns(4)
        with cols[0]:
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=75.0, step=1.0)
            dewp = st.number_input("Dew Point (°C)", min_value=-50.0, max_value=50.0, value=5.0, step=0.5)
        with cols[1]:
            temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=10.0, step=0.5)
            pres = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1015.0, step=1.0)
        with cols[2]:
            wind_dir = st.selectbox("Wind Direction", ["South-East (SE)", "Calm/Variable (cv)", "North-East (NE)", "North-West (NW)"])
            iws = st.number_input("Cumulated Wind Speed (m/s)", min_value=0.0, value=20.0, step=1.0)
        with cols[3]:
            snow = st.number_input("Cumulated Snow (hours)", min_value=0.0, value=0.0, step=1.0)
            rain = st.number_input("Cumulated Rain (hours)", min_value=0.0, value=0.0, step=1.0)

        # Center the button
        _, col_btn, _ = st.columns([2,1,2])
        with col_btn:
             predict_button = st.button("Forecast Next Hour 🔮")
    
    st.markdown("---")

    # --- Prediction Logic and Display ---
    if predict_button:
        # 1. Create the input array for the current hour, matching training order
        wind_map = {"North-East (NE)": [1,0,0,0], "North-West (NW)": [0,1,0,0], "South-East (SE)": [0,0,1,0], "Calm/Variable (cv)": [0,0,0,1]}
        # Order must be: pm2.5, DEWP, TEMP, PRES, Iws, Is, Ir, and then the 4 wind values in the same order as training
        # We assume the order of wind columns is cv, NE, NW, SE
        current_data = np.array([
            pm25, dewp, temp, pres, iws, snow, rain,
            1 if wind_dir == "Calm/Variable (cv)" else 0,
            1 if wind_dir == "North-East (NE)" else 0,
            1 if wind_dir == "North-West (NW)" else 0,
            1 if wind_dir == "South-East (SE)" else 0,
        ])

        # 2. Construct the full sequence for prediction
        input_sequence_unscaled = np.vstack([sample_history[1:], current_data])
        
        # 3. Scale the sequence
        input_sequence_scaled = scaler.transform(input_sequence_unscaled)
        
        # 4. Reshape for LSTM and predict
        reshaped_input = input_sequence_scaled.reshape(1, LOOK_BACK, EXPECTED_FEATURES)
        with st.spinner('🧠 Forecasting with LSTM...'):
            prediction_scaled = model.predict(reshaped_input)

        # 5. Inverse transform the prediction
        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[0, 0] = prediction_scaled[0, 0]
        prediction = scaler.inverse_transform(dummy_array)[0, 0]
        
        # 6. Display results
        st.header("📈 Forecast Result")
        category, color = get_aqi_category(prediction)
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid black;">
            <h3 style="color: black; margin:0;">Predicted PM2.5: {prediction:.2f} µg/m³</h3>
            <h2 style="color: black; margin:0;">Category: {category}</h2>
        </div>
        """, unsafe_allow_html=True)
else:
    st.error("Model files not found. Please run `train.py` to generate the model and scaler.")
    st.info("Ensure `lstm_aqi_model.h5` and `scaler.pkl` are in the same directory as this app.")

