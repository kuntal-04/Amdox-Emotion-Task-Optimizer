import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Page Config

st.set_page_config(page_title="Emotion & Task Analytics Dashboard", layout="wide")

st.title("AI-Powered Emotion & Task Recommendation Dashboard")
st.write("Internship Project – Amdox Technologies")

# Load Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train.txt")

data = pd.read_csv(
    DATA_PATH,
    sep=";",
    header=None,
    names=["text", "emotion"],
    engine="python",
    encoding="utf-8"
)

emotion_map = {
    "joy": "happy",
    "anger": "angry",
    "sadness": "stressed",
    "fear": "stressed",
    "love": "motivated",
    "surprise": "neutral"
}

data["emotion"] = data["emotion"].map(emotion_map)
data = data.dropna()

# Train Model 
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X = vectorizer.fit_transform(data["text"])
y = data["emotion"]

model = MultinomialNB()
model.fit(X, y)

# Task Recommendation Function
def recommend_task(emotion):
    return {
        "happy": "Creative or collaborative tasks",
        "motivated": "High priority or challenging tasks",
        "neutral": "Routine tasks",
        "stressed": "Low pressure or support tasks",
        "angry": "Break or counseling support"
    }.get(emotion, "General tasks")

# Dashboard Layout

col1, col2 = st.columns(2)

# -------- Emotion Detection  --------
with col1:
    st.subheader("Text-based Emotion Detection")
    user_text = st.text_area("Enter employee feedback:")

    if st.button("Analyze Emotion"):
        vec = vectorizer.transform([user_text])
        pred_emotion = model.predict(vec)[0]
        task = recommend_task(pred_emotion)

        st.success(f"Detected Emotion: {pred_emotion}")
        st.info(f"Recommended Task: {task}")

# Analytics Section 
st.subheader("Mood & Task Analytics")

col3, col4 = st.columns(2)

with col3:
    st.image(os.path.join(BASE_DIR, "..", "emotion_distribution.png"),
             caption="Emotion Distribution")

with col4:
    st.image(os.path.join(BASE_DIR, "..", "task_recommendation_distribution.png"),
             caption="Task Recommendation Distribution")

# Stress Alert Section 
st.subheader("Stress Monitoring")

stress_ratio = (data["emotion"] == "stressed").mean() * 100

st.write(f"Stress Level in Dataset: {stress_ratio:.2f}%")

if stress_ratio > 40:
    st.warning("High stress levels detected. Management intervention recommended.")
else:
    st.success("Stress levels are within acceptable range.")

# Privacy Section 
st.subheader("Data Privacy & Ethics")

st.markdown("""
- Dataset used is anonymized  
- No personal identifiers stored  
- No real-time user data captured  
- Local processing ensures privacy  
""")
