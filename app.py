import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# --- Constants ---
# Updated paths to look for files in the root directory
MODEL_PATH = 'imdb_sentiment_lstm.h5'
WORD_INDEX_PATH = 'word_index.json'
MAX_LEN = 200  # This must match the training configuration

# --- Load Model and Word Index ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model_and_word_index():
    """Loads the saved Keras model and the word_index dictionary."""
    print("Loading model and word index from root directory...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(WORD_INDEX_PATH, 'r') as f:
            word_index = json.load(f)
        print("Loading complete.")
        return model, word_index
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.error(
            "Please make sure 'imdb_sentiment_lstm.h5' and "
            "'word_index.json' are in the same directory as app.py."
        )
        return None, None

model, word_index = load_model_and_word_index()

# --- Preprocessing Function ---
def encode_and_pad_text(text, word_idx, max_len):
    """Encodes a text review into a padded sequence of integers."""
    words = text.lower().split()
    # Use word_idx.get(word, 2) to handle unknown words (2 is the index for <UNK>)
    encoded_review = [word_idx.get(word, 2) for word in words]
    padded_review = pad_sequences(
        [encoded_review], maxlen=max_len, padding='post', truncating='post')
    return padded_review

# --- Streamlit UI ---
st.set_page_config(page_title="Movie Review Sentiment Analyzer",
                   layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown(
    "Welcome from Surat, Gujarat! This app uses an LSTM model to predict "
    "if a movie review is **Positive** or **Negative**."
)
st.markdown("---")

user_input = st.text_area("Enter your movie review here:",
                          height=150,
                          placeholder="e.g., The movie was fantastic! The acting was superb and I loved the story...")

if st.button("Analyze Sentiment"):
    if model is not None and word_index is not None:
        if user_input:
            # 1. Preprocess the input
            processed_input = encode_and_pad_text(user_input, word_index, MAX_LEN)

            # 2. Make prediction
            with st.spinner('Analyzing...'):
                prediction_score = model.predict(processed_input, verbose=0)[0][0]

            # 3. Display result
            if prediction_score > 0.5:
                st.success(f"**Sentiment: Positive** (Confidence: {prediction_score:.2%})")
                st.balloons()
            else:
                st.error(f"**Sentiment: Negative** (Confidence: {1 - prediction_score:.2%})")
        else:
            st.warning("Please enter a review to analyze.")
    else:
        st.error("Model is not loaded. Cannot perform analysis.")

st.markdown("---")
st.info("This web app is powered by Streamlit and a TensorFlow/Keras LSTM model.")

