import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 0;
        }
        .subtext {
            text-align: center;
            font-size: 18px;
            color: #777777;
            margin-bottom: 30px;
        }
        .prediction-box {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #007acc;
            background-color: #e8f0fe;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and instructions
st.markdown('<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Enter a review below and find out whether the sentiment is positive or negative using an LSTM model trained on movie reviews.</div>', unsafe_allow_html=True)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@\\w+|#|[^a-zA-Z\\s]", "", text)
    text = re.sub(r"\\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load model/tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_model_best.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

# Text input
user_input = st.text_area("✏️ Write your movie review here:", height=200)

# Predict button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        model = load_model()
        tokenizer = load_tokenizer()

        cleaned = preprocess_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

        prediction = model.predict(padded)
        sentiment = "Positive 😀" if prediction[0][0] > 0.5 else "Negative 😞"

        st.markdown(f'<div class="prediction-box">Predicted Sentiment: {sentiment}</div>', unsafe_allow_html=True)
