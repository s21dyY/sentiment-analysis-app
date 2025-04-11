import numpy as np
import pickle
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load tokenizer and model once
with open("src/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("models/sentiment_lstm_model.h5")

# Must match training setting
max_length = 200

def preprocess_text(text):
    text = re.sub(r'[^a-z\s]', '', text) 
    return text.lower().strip()

def predict_sentiment(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0][0]  # get the probability
    label = "Positive" if prediction >= 0.5 else "Negative"
    return f"{label} ({prediction:.2f})"
