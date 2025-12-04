import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# CLEAN TEXT FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# LOAD DATA
df = pd.read_csv("data/spam_large_dataset_500.csv")
s
df["label_num"] = df["v1"].map({"ham": 0, "spam": 1})
df["clean_text"] = df["v2"].apply(clean_text)

# TRAIN TEST SPLIT
X = df["clean_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN MODEL PIPELINE
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
    ("clf", MultinomialNB())
])

model.fit(X_train, y_train)

# SAVE MODEL
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")

print("Model training complete! Saved to model/spam_model.pkl")
