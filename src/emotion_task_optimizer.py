
# Safe non-gui setup for Matplotlib
import matplotlib
matplotlib.use("Agg")

# Import libraries 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from cv_emotion_detector import detect_face_emotion


# Loading dataset from system path
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

print("Dataset loaded")
print(data.head())
print("Total rows:", len(data))


# Cleaning the dataset
data["text"] = data["text"].astype(str)
data = data[data["text"].str.strip() != ""]


# Emotion mapping and filtering
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

print("\nEmotion counts:")
print(data["emotion"].value_counts())

if data.shape[0] < 20:
    raise RuntimeError("Dataset too small after cleaning.")


# Emotion distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x="emotion", data=data)
plt.title("Emotion Distribution")
plt.tight_layout()
plt.savefig("emotion_distribution.png")
plt.close()


# TF-IDF Vectorization
X = data["text"].values
y = data["emotion"].values

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=1,          
    max_features=3000
)

X_vec = vectorizer.fit_transform(X)


# Training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)


# Evaluation on test set
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix plotting
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# Task recommendation based on emotion
def recommend_task(emotion):
    return {
        "happy": "Creative or collaborative tasks",
        "motivated": "High priority or challenging tasks",
        "neutral": "Routine tasks",
        "stressed": "Low pressure or support tasks",
        "angry": "Break or counseling support"
    }.get(emotion, "General tasks")



# Task Recommendation Distribution Plot
data["recommended_task"] = data["emotion"].apply(recommend_task)

plt.figure(figsize=(6, 6))
data["recommended_task"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Task Recommendation Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("task_recommendation_distribution.png")
plt.close()



# Final Testing with sample text
test_text = ["I feel grouchy!"]
test_vec = vectorizer.transform(test_text)

pred_emotion = model.predict(test_vec)[0]
task = recommend_task(pred_emotion)

print("\nPredicted Emotion:", pred_emotion)
print("Recommended Task:", task)



# Emotion Detection from Image using CV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "..", "images", "employee_face.jpg")

visual_emotion = detect_face_emotion(image_path)

print("\nVisual Emotion Detected (CV):", visual_emotion)

if visual_emotion in ["happy", "motivated", "neutral", "stressed", "angry"]:
    print("Recommended Task (CV-based):", recommend_task(visual_emotion))
else:
    print("CV-based task recommendation skipped.")

