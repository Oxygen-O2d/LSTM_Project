import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# --- Constants ---
# Paths for the spam detection project artifacts
MODEL_PATH = 'spam_lstm_model.h5'
WORD_INDEX_PATH = 'spam_tokenizer_word_index.json'
MAX_LEN = 100  # This must match the training configuration

# --- Load Model and Word Index ---
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the saved Keras model and the tokenizer's word_index."""
    print("Loading spam detection model and tokenizer...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(WORD_INDEX_PATH, 'r') as f:
            word_index = json.load(f)
        print("Loading complete.")
        return model, word_index
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.error(
            "Please run 'train_spam_model.py' first to generate the necessary files."
        )
        return None, None

model, word_index = load_model_and_tokenizer()

# --- Preprocessing Function ---
def encode_and_pad_text(text, word_idx, max_len):
    """Encodes an SMS message into a padded sequence of integers."""
    words = text.lower().split()
    # Use word_idx.get(word, 1) for <OOV> token, as Keras Tokenizer usually assigns 1 for it
    encoded_message = [word_idx.get(word, 1) for word in words]
    padded_message = pad_sequences(
        [encoded_message], maxlen=max_len, padding='post', truncating='post'
    )
    return padded_message

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Detector", layout="centered")

st.title("📱 SMS Spam Detector")
st.markdown(
    "Hello from Surat, Gujarat! This app uses an LSTM model to classify SMS messages as **Spam** or **Ham** (not spam)."
)
st.markdown("---")

user_input = st.text_area(
    "Enter an SMS message to classify:",
    height=100,
    placeholder="e.g., Congratulations! you have won a $1,000 Walmart gift card. Go to http://..."
)

if st.button("Classify Message"):
    if model is not None and word_index is not None:
        if user_input:
            # 1. Preprocess the input
            processed_input = encode_and_pad_text(user_input, word_index, MAX_LEN)

            # 2. Make prediction
            with st.spinner('Analyzing message...'):
                prediction_score = model.predict(processed_input, verbose=0)[0][0]

            # 3. Display result
            # 0 = Ham, 1 = Spam
            if prediction_score > 0.5:
                st.error(f"**Classification: SPAM** (Confidence: {prediction_score:.2%})")
            else:
                st.success(f"**Classification: HAM** (Confidence: {1 - prediction_score:.2%})")
        else:
            st.warning("Please enter a message to classify.")
    else:
        st.error("Model is not loaded. Cannot perform analysis.")

st.markdown("---")
st.info("Powered by Streamlit and a TensorFlow/Keras LSTM model.")

