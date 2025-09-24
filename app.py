import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import os
import time

# --- Configuration ---
MODEL_FILE = 'lstm_aqi_model.h5'
SCALER_FILE = 'scaler.pkl'
HISTORY_FILE = 'sample_history.npy' # New file for realistic history
LOOK_BACK = 24  # Must be the same as used in training
EXPECTED_FEATURES = 11 

# --- Helper Functions ---

def get_aqi_category(pm25):
    """Returns the AQI category and color for a given PM2.5 value."""
    if pm25 is None or np.isnan(pm25):
        return "Invalid", "#808080", "white" # Category, background, text color
    if 0 <= pm25 <= 50:
        return "Good", "#00e400", "black"
    elif 51 <= pm25 <= 100:
        return "Moderate", "#ffff00", "black"
    elif 101 <= pm25 <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "black"
    elif 151 <= pm25 <= 200:
        return "Unhealthy", "#ff0000", "white"
    elif 201 <= pm25 <= 300:
        return "Very Unhealthy", "#8f3f97", "white"
    else:
        return "Hazardous", "#7e0023", "white"

# --- Load Model and Scaler ---
@st.cache_resource
def load_prediction_assets():
    """Loads the trained LSTM model, scaler, and sample history."""
    files_exist = all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, HISTORY_FILE])
    if not files_exist:
        return None, None, None
        
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        sample_history = np.load(HISTORY_FILE, allow_pickle=True)
        
        # --- Validation Check ---
        if scaler.n_features_in_ != EXPECTED_FEATURES or sample_history.shape[1] != EXPECTED_FEATURES:
            st.error(
                "Asset mismatch! The scaler or sample history file doesn't match the app's configuration. "
                "Please retrain the model with the updated 'train.py' script."
            )
            return None, None, None

        return model, scaler, sample_history
    except Exception as e:
        st.error(f"Error loading prediction assets: {e}")
        return None, None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="AQI Forecaster", layout="wide", initial_sidebar_state="collapsed")

# Load assets
model, scaler, sample_history = load_prediction_assets()

# Custom CSS for the new "glassmorphism" design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: white;
    }
    .stApp {
         background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    }
    .stNumberInput > div > div > input {
        color: white;
    }

    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
    }

    .stButton>button {
        background-image: linear-gradient(to right, #DA22FF 0%, #9733EE  51%, #DA22FF  100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 15px 0;
        width: 100%;
        transition: 0.5s;
        background-size: 200% auto;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:hover {
        background-position: right center;
        box-shadow: 0 0 20px #9733EE;
    }
    h1, h2, h3, .stMarkdown {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: transparent;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Main App Logic ---
if model is not None:
    st.title("🌬️ AQI Forecaster LSTM")
    st.markdown("<p style='font-size: 1.2rem; color: #a0aec0;'>A deep learning model to predict PM2.5 air pollution for the next hour.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # --- User Input Section ---
    with st.container():
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
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
            snow = st.number_input("Cumulated Snow (hours)", min_value=0.0, value=0.0, step=1.0, key="Is")
            rain = st.number_input("Cumulated Rain (hours)", min_value=0.0, value=0.0, step=1.0, key="Ir")

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("") # Spacer
        _, col_btn, _ = st.columns([2, 1, 2])
        with col_btn:
             predict_button = st.button("Forecast Next Hour")
    
    st.markdown("---")

    # --- Prediction Logic and Animated Display ---
    if predict_button:
        # Create the input array for the current hour
        current_data = np.array([
            pm25, dewp, temp, pres, iws, snow, rain,
            1 if wind_dir == "Calm/Variable (cv)" else 0,
            1 if wind_dir == "North-East (NE)" else 0,
            1 if wind_dir == "North-West (NW)" else 0,
            1 if wind_dir == "South-East (SE)" else 0,
        ])

        # Construct, scale, and predict
        input_sequence_unscaled = np.vstack([sample_history[1:], current_data])
        input_sequence_scaled = scaler.transform(input_sequence_unscaled)
        reshaped_input = input_sequence_scaled.reshape(1, LOOK_BACK, EXPECTED_FEATURES)
        
        with st.spinner('🧠 LSTM model is thinking...'):
            prediction_scaled = model.predict(reshaped_input)

        # Inverse transform the prediction
        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[0, 0] = prediction_scaled[0, 0]
        prediction = scaler.inverse_transform(dummy_array)[0, 0]
        prediction = max(0.0, prediction) # Safeguard
        
        # Display results with animation
        st.header("📈 Forecast Result")
        
        with st.container():
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            res_cols = st.columns([1, 1.5])
            with res_cols[0]:
                category, bg_color, text_color = get_aqi_category(prediction)
                
                # Animated number count-up
                result_placeholder = st.empty()
                display_value = 0.0
                step = prediction / 50.0
                for i in range(51):
                    display_value = min(prediction, i * step)
                    result_placeholder.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; text-align: center;">
                            <p style="color: {text_color}; font-size: 1.2rem; margin:0;">Predicted PM2.5</p>
                            <h2 style="color: {text_color}; font-size: 3rem; margin:0;">{display_value:.2f}</h2>
                            <p style="color: {text_color}; font-size: 1.5rem; margin:0;">({category})</p>
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.02)

            with res_cols[1]:
                # Plotly Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={'text': "PM2.5 AQI Scale", 'font': {'color': 'white'}},
                    number={'font': {'color': 'white'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 301], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont':{'color':'white'}},
                        'bar': {'color': bg_color},
                        'steps': [
                            {'range': [0, 50], 'color': '#00e400'},
                            {'range': [51, 100], 'color': '#ffff00'},
                            {'range': [101, 150], 'color': '#ff7e00'},
                            {'range': [151, 200], 'color': '#ff0000'},
                            {'range': [201, 300], 'color': '#8f3f97'},
                            {'range': [301, 500], 'color': '#7e0023'}],
                    }))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Model assets not found. Please run `train.py` to generate `lstm_aqi_model.h5`, `scaler.pkl`, and `sample_history.npy`.")

