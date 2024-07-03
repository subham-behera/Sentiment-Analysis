import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = tf.keras.models.load_model('sentiment_model.h5')
tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Download NLTK stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def predict_sentiment(text):
    clean_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    sentiment = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
    return sentiment[0]

st.title('Sentiment Analysis App')
text_input = st.text_area('Enter Text:')
if st.button('Predict'):
    sentiment = predict_sentiment(text_input)
    st.write(f'Sentiment: {sentiment}')
