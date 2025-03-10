import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Custom Styling (Changes text area label color)
st.markdown(
    """
    <style>
        html, body, [class*="st-"] {
            background-color: white !important;
            color: black !important;
        }
        label[data-testid="stTextAreaLabel"] {
            color: blue !important;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display an Image (Check if File Exists)
image_path = "Innomatics-Logo1.png"
if os.path.exists(image_path):
    st.image(image_path, width=700)
else:
    st.warning("Image not found! Check the file path.")

# Streamlit UI
st.title("😊 Emotion Detection from Text")

# 1. Business and Data Understanding
st.header("📌 1. Business and Data Understanding")

# a) Business Problem
st.subheader("📍 a) Business Problem of Emotion Detection Using ML")
st.write("""
Emotion detection using machine learning aims to classify human emotions from text data. 
The goal is to develop an AI-driven system that can accurately recognize emotions from written sentences.  
This has applications in:
- **Customer Support:** Detecting customer sentiment in reviews and chat conversations.
- **Mental Health Monitoring:** Identifying distress or emotional patterns in patients.
- **Marketing & Brand Monitoring:** Understanding customer sentiment towards products and services.
- **Social Media & Content Analysis:** Identifying trends and reactions from social media posts.
""")

# b) Business Objective
st.subheader("📍 b) Business Objective")
st.write("""
The primary objective is to build a machine learning model that can classify emotions accurately based on textual input.  
The model should:
- Provide **high accuracy** in emotion classification.
- Be **efficient** for real-time or batch processing.
- Offer **interpretability** for business use cases.
""")

# c) Business Constraints
st.subheader("📍 c) Business Constraints")
st.write("""
- **Accuracy vs. Speed:** The model must balance accuracy and computational efficiency.
- **Data Privacy & Compliance:** Sensitive emotional data should be handled with care.
- **Scalability:** The system should be scalable for real-time analysis.
- **Bias & Fairness:** The model should minimize biases in emotion detection.
""")

# d) Data Understanding
st.subheader("📍 d) Data Understanding")
st.write("""
- **Dataset Size:** 422,746 text samples.
- **Columns:**
  1. **sentence (Text Feature)** – Contains textual data representing different emotional expressions.
  2. **emotion (Target Label)** – Represents the classified emotion associated with each sentence (e.g., happy, sad, fear, etc.).
""")

# Load the trained model
with open("emotion_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load CountVectorizer (required for text processing)
vectorizer = model.named_steps["BOW"]

# Project Description
st.header("🎭 About This Project")
st.write("""
This **Emotion Detection Model** analyzes text and predicts the **emotion behind the sentence**.  
It uses **Natural Language Processing (NLP)** and a **Naïve Bayes classifier** trained on labeled emotions.  
Simply enter a sentence, and the model will detect whether it expresses happiness, sadness, anger, surprise, or other emotions!  
""")

# 🧑‍💻 Load Model & Vectorizer
model_path = r"C:\Users\maren\INNOMATICS_327\ML_classes_327\PROJECTS\EMOTION_DETECTION_Project_7\emotion_detection_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Check if pipeline is correctly loaded
    if hasattr(model, "named_steps"):
        vectorizer = model.named_steps["vectorizer"]  # Make sure this is the correct name
        classifier = model.named_steps["classifier"]
    else:
        st.error("❌ Invalid model format. Ensure it's a valid scikit-learn pipeline.")
        st.stop()
else:
    st.error("❌ Model file not found! Please check your working directory.")
    st.stop()

# 📊 Load Dataset for Performance Metrics
data_path = "combined_emotion.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    X = df["sentence"]
    y = df["emotion"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform text and predict
    X_test_transformed = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_transformed)

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    st.subheader("📊 Model Performance")
    st.write(f"✅ **Accuracy:** {accuracy:.2%} 🎯")
    st.write(f"✅ **F1-Score:** {f1:.2%} 🔥")
else:
    st.warning("⚠️ Dataset not found! Performance metrics cannot be displayed.")

# 😃 Emotion Mapping with Colors
emotion_mapping = {
    "joy": ("😊", "#FFD700"),
    "sad": ("😢", "#6495ED"),
    "anger": ("😠", "#FF4500"),
    "fear": ("😨", "#A9A9A9"),
    "surprise": ("😲", "#32CD32"),
    "neutral": ("😐", "#808080"),
    "love": ("❤️", "#FF1493"),
    "disgust": ("🤢", "#8B0000")
}

# 📝 User Input
user_input = st.text_area("📝 Enter your text here:")

if st.button("🔍 Detect Emotion"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = classifier.predict(transformed_input)[0]
        emoji, color = emotion_mapping.get(prediction.lower(), ("😶", "#000"))
        
        st.markdown(
            f'<p style="color:{color}; font-size:24px; font-weight:bold;">🎭 Predicted Emotion: {prediction} {emoji}</p>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter text before predicting.")


