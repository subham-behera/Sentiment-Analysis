import streamlit as st
import joblib
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load the tokenizer and model
tokenizer = joblib.load('tokenizer.pkl')
model = load_model('model.h5')

# Define the maximum length of sequences (must match the training settings)
max_len = 200

# Define the sentiment labels
sentiment = ['Neutral', 'Negative', 'Positive']

def predict(text):
    sequence = tokenizer.texts_to_sequences([text])
    test = pad_sequences(sequence, maxlen=max_len)
    return sentiment[model.predict(test).argmax()]

# Streamlit app
st.title("Sentiment Analysis")

text = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict"):
    if text:
        result = predict(text)
        st.write(f"Sentiment: {result}")
    else:
        st.write("Please enter some text.")
