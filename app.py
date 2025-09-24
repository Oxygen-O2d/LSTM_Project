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

model, scaler = load_prediction_model()

# --- Streamlit App UI ---
st.set_page_config(page_title="Energy Consumption Prediction", layout="wide")
st.title("⚡️ Household Energy Consumption Prediction")
st.markdown(
    "This app uses an LSTM model to predict the next hour's energy consumption (in kilowatts) "
    "based on the consumption data from the previous 24 hours."
)

if model is None or scaler is None:
    st.stop()

# --- User Input Section ---
st.sidebar.header("Input Historical Data")
st.sidebar.write(f"Enter the last {LOOK_BACK} hours of Global Active Power (kW):")

# Initialize input fields in the sidebar
input_data = []
# Using a sample from a known period for realistic default values
default_values = [
    1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4,
    0.5, 0.7, 1.0, 1.5, 2.0, 2.2, 2.1, 2.0, 1.8, 1.6, 1.4, 1.3
]

for i in range(LOOK_BACK):
    default_val = default_values[i] if i < len(default_values) else 1.0
    val = st.sidebar.number_input(
        f"Hour {i+1}",
        min_value=0.0,
        max_value=10.0,
        value=default_val,
        step=0.1,
        key=f"hour_{i}"
    )
    input_data.append(val)

# --- Prediction and Display ---
if st.sidebar.button("Predict Next Hour 🚀"):
    # 1. Prepare data for prediction
    input_array = np.array(input_data).reshape(-1, 1)
    scaled_input = scaler.transform(input_array)
    reshaped_input = scaled_input.reshape(1, LOOK_BACK, 1)

    # 2. Make prediction
    prediction_scaled = model.predict(reshaped_input)
    prediction = scaler.inverse_transform(prediction_scaled)
    predicted_value = prediction[0][0]

    # 3. Display the result
    st.subheader("Prediction Result")
    st.metric(label="Predicted Consumption for the Next Hour", value=f"{predicted_value:.4f} kW")

    # 4. Visualize the input and prediction
    st.subheader("Visualization")
    # Create a DataFrame for plotting
    history_df = pd.DataFrame({
        'Hour': [f'H-{LOOK_BACK-i}' for i in range(LOOK_BACK)],
        'Consumption (kW)': input_data,
        'Type': 'Historical'
    })
    prediction_df = pd.DataFrame({
        'Hour': ['H+1 (Predicted)'],
        'Consumption (kW)': [predicted_value],
        'Type': 'Predicted'
    })
    plot_df = pd.concat([history_df, prediction_df], ignore_index=True)

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df['Hour'][:-1],
        y=plot_df['Consumption (kW)'][:-1],
        name='Historical Data',
        marker_color='skyblue'
    ))
    fig.add_trace(go.Bar(
        x=[plot_df['Hour'].iloc[-1]],
        y=[plot_df['Consumption (kW)'].iloc[-1]],
        name='Predicted Value',
        marker_color='salmon'
    ))

    fig.update_layout(
        title="Historical and Predicted Energy Consumption",
        xaxis_title="Time Horizon",
        yaxis_title="Global Active Power (kW)",
        legend_title="Data Type",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter the data in the sidebar and click 'Predict' to see the forecast.")
