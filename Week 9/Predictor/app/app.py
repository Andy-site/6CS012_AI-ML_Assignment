from flask import Flask, render_template, request
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer and model
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
model = tf.keras.models.load_model('models/best_lstm.h5')

# Use the same max_len from your notebook
max_len = 100  # <-- replace with your computed max_len value

# Preprocessing function (same as notebook)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    if request.method == 'POST':
        tweet = request.form['tweet']
        clean_tweet = preprocess_text(tweet)
        seq = tokenizer.texts_to_sequences([clean_tweet])
        pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        prob = model.predict(pad)[0][0]
        sentiment = 'Positive' if prob > 0.5 else 'Negative'
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)