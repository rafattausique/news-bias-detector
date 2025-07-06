import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob
import re

# Load the model and vectorizer
model = joblib.load("bias_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess function
def preprocess(text):
    text = re.sub(r'\W+', ' ', text)
    text = text.lower().strip()
    return text

# Predict function
def predict(text):
    clean_text = preprocess(text)
    vect_text = vectorizer.transform([clean_text])
    pred_proba = model.predict_proba(vect_text)[0]
    pred_class = model.predict(vect_text)[0]
    confidence = round(max(pred_proba) * 100, 2)
    return pred_class, confidence

# Example list
examples = [
    "Government introduces a game-changing policy for economic growth.",
    "Opposition slams ruling party over controversial education bill.",
    "New environmental law hailed as a breakthrough by activists.",
    "Experts warn of rising inflation despite central bank's optimism.",
    "Healthcare reforms criticized for favoring private corporations."
]

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="News Bias Detector", page_icon="📰")
st.title("📰 News Bias Detector")
st.markdown("Built using Machine Learning and NLP to detect bias in news headlines or short texts.")

# ---------- Get or initialize index ----------
example_index = st.session_state.get('example_index', 0)

# ---------- Text Input with current example loaded ----------
user_input = st.text_area(
    "**Enter your news text:**",
    value=examples[example_index],
    height=200
)

# ---------- Define 3-column layout: left, spacer, right ----------
col1, col_spacer, col3 = st.columns([1, 4, 1])

# ---------- Left: Show Example ----------
with col1:
    if st.button("🔁 Show an example"):
        st.session_state['example_index'] = (example_index + 1) % len(examples)
        st.rerun()

# ---------- Right: Detect Bias ----------
with col3:
    if st.button("🔍 Detect Bias"):
        if not user_input.strip():
            st.warning("Please enter or upload a news text.")
        else:
            pred, confidence = predict(user_input)
            st.success(f"**Prediction:** {pred}  \n**Confidence:** {confidence}%")

# File upload option
st.markdown("### 📁 Or upload a file:")
uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=['txt', 'csv'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.txt'):
            content = uploaded_file.read().decode("utf-8").splitlines()
            df = pd.DataFrame(content, columns=['text'])
        else:
            df = pd.read_csv(uploaded_file)

        df['cleaned'] = df['text'].astype(str).apply(preprocess)
        df['prediction'] = model.predict(vectorizer.transform(df['cleaned']))
        df['confidence (%)'] = model.predict_proba(vectorizer.transform(df['cleaned'])).max(axis=1) * 100
        st.dataframe(df[['text', 'prediction', 'confidence (%)']])
    except Exception as e:
        st.error(f"Something went wrong: {e}")
