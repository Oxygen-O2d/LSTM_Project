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
HISTORY_FILE = 'sample_history.npy' # For realistic history
LOOK_BACK = 24  # Must be the same as used in training
EXPECTED_FEATURES = 11

# --- Helper Functions ---

def get_aqi_category(pm25):
    """Returns the AQI category, color, and text color for a given PM2.5 value."""
    if pm25 is None or np.isnan(pm25):
        return "Invalid", "#e0e0e0", "black" # Category, background, text color
    if 0 <= pm25 <= 50:
        return "Good", "#d4edda", "#155724"
    elif 51 <= pm25 <= 100:
        return "Moderate", "#fff3cd", "#856404"
    elif 101 <= pm25 <= 150:
        return "Unhealthy for Sensitive Groups", "#ffeeba", "#856404"
    elif 151 <= pm25 <= 200:
        return "Unhealthy", "#f8d7da", "#721c24"
    elif 201 <= pm25 <= 300:
        return "Very Unhealthy", "#e9d8fd", "#492b7c"
    else:
        return "Hazardous", "#d6d8db", "#383d41"

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

        if scaler.n_features_in_ != EXPECTED_FEATURES or sample_history.shape[1] != EXPECTED_FEATURES:
            st.error("Asset mismatch! Please retrain the model with the updated 'train.py' script.")
            return None, None, None

        return model, scaler, sample_history
    except Exception as e:
        st.error(f"Error loading prediction assets: {e}")
        return None, None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="AQI Forecaster", layout="wide", initial_sidebar_state="collapsed")

model, scaler, sample_history = load_prediction_assets()

# Professional, clean CSS design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f6;
    }
    .main {
        background-color: #f0f2f6;
    }
    .card-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        background-color: #0068c9;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px 0;
        width: 100%;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #005aa3;
        box-shadow: 0 2px 8px rgba(0, 104, 201, 0.4);
    }
    h1, h2, h3 {
        color: #1f2937 !important;
    }
    .stMarkdown, p, .stNumberInput, .stSelectbox {
        color: #4b5563;
    }
    .prediction-container {
        display: flex;
        align-items: center; /* Vertically align items */
        justify-content: center;
        height: 350px; /* Fixed height for alignment */
    }
    .prediction-card {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        width: 100%;
        height: 100%;
        padding: 20px;
        border-radius: 10px;
        transition: background-color 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# --- Main App Logic ---
if model is not None:
    st.title("Air Quality Index Forecaster")
    st.markdown("<p style='font-size: 1.1rem; color: #4b5563;'>Predict next-hour PM2.5 concentrations using a recurrent neural network.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.container():
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.header("Current Environmental Conditions")

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

        st.write("")
        _, col_btn, _ = st.columns([2.2, 1, 2.2])
        with col_btn:
             predict_button = st.button("Generate Forecast")

    st.markdown("---")

    if predict_button:
        current_data = np.array([
            pm25, dewp, temp, pres, iws, snow, rain,
            1 if wind_dir == "Calm/Variable (cv)" else 0,
            1 if wind_dir == "North-East (NE)" else 0,
            1 if wind_dir == "North-West (NW)" else 0,
            1 if wind_dir == "South-East (SE)" else 0,
        ])

        input_sequence_unscaled = np.vstack([sample_history[1:], current_data])
        input_sequence_scaled = scaler.transform(input_sequence_unscaled)
        reshaped_input = input_sequence_scaled.reshape(1, LOOK_BACK, EXPECTED_FEATURES)

        with st.spinner('Analyzing data with LSTM model...'):
            prediction_scaled = model.predict(reshaped_input)

        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[0, 0] = prediction_scaled[0, 0]
        prediction = scaler.inverse_transform(dummy_array)[0, 0]
        prediction = max(0.0, prediction)

        st.header("Forecast Result")

        with st.container():
            st.markdown("<div class='card-container'>", unsafe_allow_html=True)
            res_cols = st.columns(2, gap="large")
            with res_cols[0]:
                category, bg_color, text_color = get_aqi_category(prediction)
                result_placeholder = st.empty()
                
                # Animate the count-up
                for i in range(51):
                    display_value = min(prediction, i * (prediction / 50.0) if prediction > 0 else 0)
                    result_placeholder.markdown(f"""
                        <div class="prediction-container">
                            <div class="prediction-card" style="background-color: {bg_color}; border: 1px solid {text_color}33;">
                                <p style="color: {text_color}; font-size: 1.1rem; margin:0; font-weight: 600;">PREDICTED PM2.5</p>
                                <h2 style="color: {text_color}; font-size: 4rem; font-weight: 700; margin:0;">{display_value:.2f}</h2>
                                <p style="color: {text_color}; font-size: 1.5rem; margin:0; font-weight: 600;">{category}</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.01)

            with res_cols[1]:
                # Plotly Gauge Chart, styled for light theme
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={'text': "AQI Scale", 'font': {'color': '#1f2937', 'size': 18}},
                    number={'font': {'color': '#1f2937', 'size': 36}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 301], 'tickwidth': 1, 'tickcolor': "#4b5563", 'tickfont':{'color':'#4b5563'}},
                        'bar': {'color': '#4b5563', 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 50], 'color': '#d4edda'}, {'range': [51, 100], 'color': '#fff3cd'},
                            {'range': [101, 150], 'color': '#ffeeba'}, {'range': [151, 200], 'color': '#f8d7da'},
                            {'range': [201, 300], 'color': '#e9d8fd'}, {'range': [301, 500], 'color': '#d6d8db'}],
                    }))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#1f2937"},
                    height=350, # Match the height of the prediction card container
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Model assets not found. Please run `train.py` to generate the required model and data files.")

