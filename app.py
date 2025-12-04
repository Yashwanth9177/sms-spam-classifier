import streamlit as st
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords

# Fix SSL for NLTK (optional)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load model
model = joblib.load("model/spam_model.pkl")

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# UI
st.title("📧 Email Spam Classifier")

email_text = st.text_area("Paste email content here:")

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(email_text)
        prediction = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0][prediction]

        if prediction == 1:
            st.error(f"🚨 SPAM (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ NOT SPAM (Confidence: {proba:.2f})")
