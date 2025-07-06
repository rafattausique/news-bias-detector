import streamlit as st
import joblib
import pandas as pd
from textblob import TextBlob
import re

# Load model and vectorizer
model = joblib.load("bias_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="News Bias Detector", layout="centered")

# Title and subtitle
st.title("📰 News Bias Detector")
st.caption("Built using Machine Learning and NLP to detect bias in news headlines or short texts.")

# Input area
st.markdown("### Enter your news text:")
input_text = st.text_area("", placeholder="e.g., Government introduces a game-changing policy for economic growth.", height=150)

# Example button
if st.button("Show an example"):
    input_text = "The new law is a blatant attack on our freedom and democracy."

# Prediction
if st.button("Detect Bias"):
    if input_text.strip():
        # Clean input
        cleaned_text = re.sub(r'[^\w\s]', '', input_text.lower())

        # Vectorize and predict
        vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max() * 100

        # Display result
        st.markdown("---")
        st.success(f"**Prediction:** {'🟥 Biased' if prediction == 1 else '🟩 Unbiased'}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")

