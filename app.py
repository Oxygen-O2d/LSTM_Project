import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

# --- Configuration ---
MODEL_FILE = 'lstm_energy_model.h5'
SCALER_FILE = 'scaler.pkl'
LOOK_BACK = 24  # Must be the same as used in training

# --- Load Model and Scaler ---
@st.cache_resource
def load_prediction_model():
    """Loads the trained LSTM model and scaler."""
    try:
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_FILE}' or scaler file '{SCALER_FILE}' not found.")
        st.info("Please run the `train.py` script first to generate these files.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="Energy Forecaster", layout="wide", initial_sidebar_state="collapsed")

# Load model and scaler
model, scaler = load_prediction_model()

# --- Main App Interface ---
if model is None or scaler is None:
    st.stop()

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 10px 0;
        font-weight: bold;
    }
    .container {
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.title("⚡️ Smart Energy Consumption Forecaster")
st.markdown(
    "An LSTM-powered tool to predict the next hour's energy consumption based on historical data. "
    "Input the last 24 hours of power usage below to generate a forecast."
)
st.markdown("---")


# --- User Input Section in a container ---
with st.container():
    st.header("📊 Input Historical Data")
    st.write(f"Enter the Global Active Power (kW) for the last {LOOK_BACK} hours.")

    # Using a sample from a known period for realistic default values
    default_values = [
        1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4,
        0.5, 0.7, 1.0, 1.5, 2.0, 2.2, 2.1, 2.0, 1.8, 1.6, 1.4, 1.3
    ]

    # Create columns for a grid layout
    cols = st.columns(6)
    input_data = []

    for i in range(LOOK_BACK):
        with cols[i % 6]:
            default_val = default_values[i] if i < len(default_values) else 1.0
            val = st.number_input(
                f"Hour {i+1}",
                min_value=0.0,
                max_value=15.0, # Increased max for wider range
                value=default_val,
                step=0.1,
                key=f"hour_{i}"
            )
            input_data.append(val)

# --- Prediction and Display ---
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict_button = st.button("Predict Next Hour 🚀", use_container_width=True)

st.markdown("---")

if predict_button:
    # 1. Prepare data for prediction
    input_array = np.array(input_data).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    reshaped_input = scaled_input.reshape(1, LOOK_BACK, 1)

    # 2. Make prediction with a spinner
    with st.spinner('🧠 LSTM model is thinking...'):
        prediction_scaled = model.predict(reshaped_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        predicted_value = prediction[0][0]

    # 3. Display the result in a styled container
    st.header("🔮 Forecast Result")
    result_col1, result_col2 = st.columns([1, 2])
    with result_col1:
        st.metric(label="Predicted Consumption for the Next Hour", value=f"{predicted_value:.4f} kW")

    # 4. Visualize the input and prediction
    with result_col2:
        # Create a DataFrame for plotting
        history_df = pd.DataFrame({
            'Hour': list(range(-LOOK_BACK + 1, 1)),
            'Consumption (kW)': input_data,
            'Type': 'Historical'
        })
        prediction_df = pd.DataFrame({
            'Hour': [1],
            'Consumption (kW)': [predicted_value],
            'Type': 'Predicted'
        })

        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['Hour'],
            y=history_df['Consumption (kW)'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='royalblue', width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=prediction_df['Hour'],
            y=prediction_df['Consumption (kW)'],
            mode='markers',
            name='Predicted Value',
            marker=dict(color='crimson', size=14, symbol='star')
        ))

        fig.update_layout(
            title="Historical vs. Predicted Energy Consumption",
            xaxis_title="Time (Hours from Now)",
            yaxis_title="Global Active Power (kW)",
            legend_title="Data Type",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey'),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter the data above and click the 'Predict' button to generate a forecast.")

