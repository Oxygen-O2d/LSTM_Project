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
HISTORY_FILE = 'sample_history.npy'
LOOK_BACK = 24
EXPECTED_FEATURES = 11

# --- Helper Functions ---
def get_aqi_category(pm25):
    """Returns the AQI category and a professional accent color."""
    if pm25 is None or np.isnan(pm25):
        return "Invalid", "#6b7280"  # Gray
    if 0 <= pm25 <= 50:
        return "Good", "#22c55e"  # Green
    elif 51 <= pm25 <= 100:
        return "Moderate", "#facc15"  # Yellow
    elif 101 <= pm25 <= 150:
        return "Unhealthy for Sensitive Groups", "#f97316"  # Orange
    elif 151 <= pm25 <= 200:
        return "Unhealthy", "#ef4444"  # Red
    elif 201 <= pm25 <= 300:
        return "Very Unhealthy", "#a855f7"  # Purple
    else:
        return "Hazardous", "#7e22ce"  # Dark Purple

# --- Asset Loading ---
@st.cache_resource
def load_prediction_assets():
    """Loads and validates the trained model, scaler, and sample history."""
    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, HISTORY_FILE]):
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

# --- UI Setup ---
st.set_page_config(page_title="AQI Forecaster", layout="wide", initial_sidebar_state="collapsed")
model, scaler, sample_history = load_prediction_assets()

# --- Professional Dark Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main {
        background-color: #0d1117;
    }
    .card-container {
        background-color: #161b22;
        border-radius: 10px;
        padding: 2rem;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #30363d;
        padding: 12px 0;
        width: 100%;
        transition: background-color 0.3s ease, border-color 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    h1, h2, h3 { color: #f0f6fc !important; }
    .stMarkdown, p, .stNumberInput, .stSelectbox { color: #c9d1d9; }
    .prediction-card {
        background-color: #161b22;
        border-left: 5px solid; /* The accent color goes here */
        border-radius: 10px;
        padding: 1.5rem 2rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .stSpinner > div {
        border-top-color: #238636 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Application Logic ---
if model is not None:
    st.title("Air Quality Index Forecaster")
    st.markdown("<p style='font-size: 1.1rem; color: #8b949e;'>Predict next-hour PM2.5 concentrations using a recurrent neural network.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.container():
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.header("Current Environmental Conditions")
        cols = st.columns(4)
        with cols[0]:
            pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, value=75.0, step=1.0)
            dewp = st.number_input("Dew Point (°C)", -50.0, 50.0, value=5.0, step=0.5)
        with cols[1]:
            temp = st.number_input("Temperature (°C)", -50.0, 50.0, value=10.0, step=0.5)
            pres = st.number_input("Pressure (hPa)", 900.0, 1100.0, value=1015.0, step=1.0)
        with cols[2]:
            wind_dir = st.selectbox("Wind Direction", ["South-East (SE)", "Calm/Variable (cv)", "North-East (NE)", "North-West (NW)"])
            iws = st.number_input("Cumulated Wind Speed (m/s)", 0.0, value=20.0, step=1.0)
        with cols[3]:
            snow = st.number_input("Cumulated Snow (hours)", 0.0, value=0.0, step=1.0, key="Is")
            rain = st.number_input("Cumulated Rain (hours)", 0.0, value=0.0, step=1.0, key="Ir")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        _, col_btn, _ = st.columns([2.2, 1, 2.2])
        if col_btn.button("Generate Forecast"):
            
            with st.spinner('Analyzing data with LSTM model...'):
                current_data = np.array([
                    pm25, dewp, temp, pres, iws, snow, rain,
                    1 if wind_dir == "Calm/Variable (cv)" else 0,
                    1 if wind_dir == "North-East (NE)" else 0,
                    1 if wind_dir == "North-West (NW)" else 0,
                    1 if wind_dir == "South-East (SE)" else 0,
                ])
                input_sequence = np.vstack([sample_history[1:], current_data])
                scaled_input = scaler.transform(input_sequence)
                reshaped_input = scaled_input.reshape(1, LOOK_BACK, EXPECTED_FEATURES)
                
                prediction_scaled = model.predict(reshaped_input)
                
                dummy_array = np.zeros((1, scaler.n_features_in_))
                dummy_array[0, 0] = prediction_scaled[0, 0]
                prediction = scaler.inverse_transform(dummy_array)[0, 0]
                prediction = max(0.0, prediction)

            st.markdown("---")
            st.header("Forecast Result")

            with st.container():
                st.markdown("<div class='card-container'>", unsafe_allow_html=True)
                res_cols = st.columns([0.8, 1.2], gap="large")
                
                with res_cols[0]:
                    category, accent_color = get_aqi_category(prediction)
                    result_placeholder = st.empty()
                    # Animate the count-up
                    for i in range(51):
                        display_value = min(prediction, i * (prediction / 50.0) if prediction > 0 else 0)
                        result_placeholder.markdown(f"""
                            <div class="prediction-card" style="border-left-color: {accent_color};">
                                <p style="color: #8b949e; font-size: 1.1rem; margin:0; font-weight: 600;">PREDICTED PM2.5</p>
                                <h2 style="color: {accent_color}; font-size: 4rem; font-weight: 700; margin:0; line-height: 1.2;">{display_value:.2f}</h2>
                                <p style="color: #c9d1d9; font-size: 1.5rem; margin:0; font-weight: 600;">{category}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.01)

                with res_cols[1]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prediction,
                        title={'text': "AQI Scale", 'font': {'color': '#f0f6fc', 'size': 18}},
                        number={'font': {'color': '#f0f6fc', 'size': 36}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 301], 'tickwidth': 1, 'tickcolor': "#8b949e", 'tickfont':{'color':'#8b949e'}},
                            'bar': {'color': '#8b949e', 'thickness': 0.3},
                            'steps': [
                                {'range': [0, 50], 'color': '#1a3c26'}, {'range': [51, 100], 'color': '#4d3c15'},
                                {'range': [101, 150], 'color': '#522b11'}, {'range': [151, 200], 'color': '#541d1d'},
                                {'range': [201, 300], 'color': '#3c2155'}, {'range': [301, 500], 'color': '#31194a'}],
                        }))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#f0f6fc"},
                        height=300,
                        margin=dict(l=10, r=10, t=60, b=10) # Increased top margin
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Model assets not found. Please run `train.py` to generate the required model and data files.")

