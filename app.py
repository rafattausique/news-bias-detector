
import streamlit as st
import os
os.system("pip install joblib")
import joblib
import pandas as pd
from textblob import TextBlob
import re
import string

# Load model and vectorizer
model = joblib.load("bias_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Bias label decoder
label_map = {
    "left": "🟥 Left-Leaning Bias",
    "center": "🟨 Neutral / Center",
    "right": "🟦 Right-Leaning Bias"
}

# App interface
st.set_page_config(page_title="News Bias Detector", layout="centered")
st.title("📰 Indian News Media Bias Detector")
st.markdown("Paste a news article or snippet to detect its **media bias**.")

# Input
user_input = st.text_area("Enter News Article", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean & vectorize
        clean = clean_text(user_input)
        vect = vectorizer.transform([clean])

        # Predict
        prediction = model.predict(vect)[0]

        # Sentiment
        sentiment = TextBlob(user_input).sentiment

        # Display results
        st.subheader("🧠 Prediction")
        st.success(f"**Bias Detected:** {label_map.get(prediction, prediction)}")
        
        st.subheader("💬 Sentiment")
        st.info(f"**Polarity:** {sentiment.polarity:.2f} | **Subjectivity:** {sentiment.subjectivity:.2f}")

        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit, scikit-learn, and TextBlob")
