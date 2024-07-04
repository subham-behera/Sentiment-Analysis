# Sentiment Analysis Deep Learning Model

This project is a Streamlit web application for predicting the sentiment of a given statement using a deep learning model.

## Project Overview

This project aims to analyze the sentiment of a given statement, classifying it as positive, negative, or neutral using a deep learning model deployed on a Streamlit app. The model is trained on a dataset of text labeled with their respective sentiment.

## Files in the Repository

- **`index.py`**: Main file to run the Streamlit app.
- **`model.pkl`**: Pre-trained sentiment analysis model saved using joblib.
- **`requirements.txt`**: List of Python packages required to run the app.
- **`README.md`**: Project documentation.
- **`train.csv`**: The dataset and any other necessary data files.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/subham-behera/sentiment-analysis.git
    cd sentiment-analysis
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

To run the Streamlit app, execute the following command in your terminal:
```bash
streamlit run app.py
