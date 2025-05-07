import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False

# Load the model and vectorizer
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocessing function (same as in training)
def preprocess(text):
    if not isinstance(text, str):
        return ""
        
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags (for social media text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    try:
        # Tokenize text into words
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        # Fallback to simple tokenization if NLTK fails
        tokens = text.split()
        return ' '.join([word for word in tokens if len(word) > 2])

# Function to predict sentiment
def predict_sentiment(text, model, vectorizer):
    # Preprocess the input text
    processed_input = preprocess(text)
    
    # Transform the input text using the vectorizer
    input_vec = vectorizer.transform([processed_input])
    
    # Get prediction probabilities
    probabilities = None
    try:
        probabilities = model.predict_proba(input_vec)[0]
        # Determine prediction based on highest probability instead of model.predict()
        prediction = probabilities.argmax()
    except:
        # Fallback to direct prediction if probabilities aren't available
        prediction = model.predict(input_vec)[0]
    
    return prediction, probabilities, processed_input

# Map sentiment codes to labels and colors
def get_sentiment_display(prediction):
    sentiment_map = {
        0: {"label": "Negative", "emoji": "😞", "color": "#FF5252"},
        1: {"label": "Neutral", "emoji": "😐", "color": "#FFC107"},
        2: {"label": "Positive", "emoji": "😊", "color": "#4CAF50"}
    }
    
    # Handle binary classification (0=negative, 1=positive)
    if prediction not in sentiment_map:
        if prediction == 1:
            return {"label": "Positive", "emoji": "😊", "color": "#4CAF50"}
        else:
            return {"label": "Negative", "emoji": "😞", "color": "#FF5252"}
    
    return sentiment_map[prediction]

# Set up the Streamlit page
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Download NLTK resources
resources_loaded = download_nltk_resources()

# Main content
st.title("✨ Sentiment Analysis Web App")
st.markdown("This app predicts the sentiment of your text as positive, negative, or neutral.")

# Sidebar with model info
with st.sidebar:
    st.header("About this App")
    st.markdown("""
    This intelligent sentiment analysis tool uses machine learning to determine the emotional tone of text. 
    
    ### How it works:
    1. Enter your text in the input box
    2. Click 'Analyze Sentiment'
    3. View the predicted sentiment and confidence scores
    
    ### Model Information:
    - Uses TF-IDF vectorization for text feature extraction
    - Trained on a labeled dataset of text with sentiment annotations
    - Supports detection of positive, negative, and neutral sentiment
    """)
    
    st.markdown("---")
    st.markdown("### Examples to try:")
    examples = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "The service was okay, but could have been better.",
        "This is the worst experience I've ever had. Don't waste your money!"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.user_input = example

# Load model resources
model, vectorizer = load_model_resources()

# Check if resources are loaded
if not resources_loaded or model is None or vectorizer is None:
    st.error("⚠️ Failed to load required resources. Please check the error messages above.")
    st.stop()

# User input section
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_area(
    "Enter text for sentiment analysis:",
    value=st.session_state.user_input,
    height=150
)

# Analysis section
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Analyze Sentiment", type="primary"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                # Get prediction
                prediction, probabilities, processed_text = predict_sentiment(user_input, model, vectorizer)
                
                # Get display information
                sentiment_info = get_sentiment_display(prediction)
                
                # Display result in a success box
                st.markdown("### Results")
                st.markdown(f"<div style='padding: 20px; border-radius: 10px; background-color: {sentiment_info['color']}30;'>"\
                            f"<h2 style='margin:0; color: {sentiment_info['color']};'>{sentiment_info['emoji']} {sentiment_info['label']}</h2>"\
                            "</div>", unsafe_allow_html=True)
                
                # Show probabilities if available
                if probabilities is not None:
                    st.markdown("### Confidence Scores")
                    labels = ["Negative", "Neutral", "Positive"] if len(probabilities) == 3 else ["Negative", "Positive"]
                    
                    for i, label in enumerate(labels):
                        # Get the color for this sentiment
                        info = get_sentiment_display(i)
                        
                        # Display progress bar
                        st.markdown(f"**{label}:**")
                        st.progress(probabilities[i])
                        st.markdown(f"<small>{probabilities[i]*100:.1f}%</small>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")

with col2:
    if user_input:
        st.markdown("### Text Analysis")
        
        # Word count
        words = user_input.split()
        st.metric("Word Count", len(words))
        
        # Char count
        st.metric("Character Count", len(user_input))
        
        # Show preprocessed text
        with st.expander("View Preprocessed Text"):
            processed = preprocess(user_input)
            st.text(processed)
            st.markdown(f"*{len(processed.split())} words after preprocessing*")

# Add footer
st.markdown("---")
st.markdown(
    """<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and scikit-learn</p>
    </div>""", 
    unsafe_allow_html=True
)