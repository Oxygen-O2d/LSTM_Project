import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- Configuration ---
MODEL_PATH = 'spam_lstm_model.h5'
TOKENIZER_PATH = 'spam_tokenizer.json' # <-- Updated path for the new tokenizer file
MAX_LEN = 100

# --- Caching Functions for Efficiency ---
@st.cache_resource
def load_model_artifacts():
    """
    Loads the trained model and tokenizer from disk.
    The @st.cache_resource decorator ensures this is only done once.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        st.error(f"Error: Model or tokenizer file not found. "
                 f"Please run the `train_spam_model.py` script first to generate '{MODEL_PATH}' and '{TOKENIZER_PATH}'.")
        st.stop()
    
    try:
        print("Loading Keras model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        print("Loading Keras tokenizer...")
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

# --- Preprocessing ---
def preprocess_input(text, tokenizer, max_len):
    """
    Takes raw text and preprocesses it using the loaded tokenizer.
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequence

# --- Load artifacts ---
model, tokenizer = load_model_artifacts()

# --- Streamlit UI ---
st.set_page_config(page_title="Spam SMS Detector", layout="centered")

st.title("🛡️ Spam SMS Detector")
st.write("Enter an SMS message below to classify it as Spam or Ham (Not Spam).")

user_input = st.text_area("Enter message here:", height=150, placeholder="e.g., Congratulations! You've won a prize...")

if st.button("Classify Message"):
    if user_input:
        with st.spinner('Analyzing message...'):
            # Preprocess the input
            processed_input = preprocess_input(user_input, tokenizer, MAX_LEN)
            
            # Make prediction
            try:
                prediction_score = model.predict(processed_input, verbose=0)[0][0]
                
                # Display result
                if prediction_score > 0.5:
                    st.error(f"Prediction: SPAM (Confidence: {prediction_score:.2%})")
                else:
                    st.success(f"Prediction: HAM (Confidence: {1 - prediction_score:.2%})")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning("Please enter a message to classify.")

st.markdown("---")
st.info("This app uses an LSTM model trained on the SMS Spam Collection Dataset.")

