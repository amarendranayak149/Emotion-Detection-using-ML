import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Custom Styling for Background & UI
# Custom Styling for Background & UI
st.markdown(
    """
    <style>
        /* Title Styling */
        .subheader {
            font-size: 30px;
            font-weight: bold;
            color: #ff5733;
            margin-top: 20px;
        }
        /* Info Box Styling */
        .info-box {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 2px 20px rgba(0,0,0,0.2);
            margin: 15px 0;
            text-align: justify;
            font-size: 18px;
            line-height: 1.6;
        }
        /* Bullet Points Styling */
        .info-box ul {
            padding-left: 20px;
        }
        /* Icon Styling */
        .emoji {
            font-size: 22px;
            margin-right: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True  # Corrected placement
)
st.markdown(
    """
    <style>
        /* Gradient Background */
        html, body, [class*="st-"] {
            background: linear-gradient(120deg, #FFDEE9, #B5FFFC) !important;
            color: black !important;
        }
        
        /* Customizing text areas */
        label[data-testid="stTextAreaLabel"] {
            color: #002855 !important;
            font-size: 20px;
        }
        /* Title Styling */
        .title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #ff5733;
        }
        /* Card-Like Sections */
        .info-box {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 2px 2px 20px rgba(0,0,0,0.2);
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Display an Image
image_path = "Innomatics-Logo1.png"
if os.path.exists(image_path):
    st.image(image_path, width=700)
else:
    st.warning("⚠ Image not found! Check the file path.")

# Streamlit UI
st.markdown('<h1 class="title">🎭 Emotion Detection from Text</h1>', unsafe_allow_html=True)

# About the Project
st.markdown('<h3 class="subheader">📌 About the Project</h3>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-box">
        This project is designed to detect emotions from textual data using machine learning.  
        By analyzing words and sentence structures, the model predicts emotions such as 😊 happiness, 😢 sadness, 😠 anger, and more.  
        This can be beneficial for various applications, including 📊 customer sentiment analysis, ❤ mental health monitoring, and 🔍 content moderation.
    </div>
    """,
    unsafe_allow_html=True
)

# Business Problem
st.markdown('<h3 class="subheader">💡 Business Problem: Emotion Detection Using AI</h3>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-box">
        In an era driven by data and emotions, understanding human sentiments from text is game-changing!  
        Imagine an AI that "feels" what you're saying—here's how it helps:
        <ul>
            <li>🤖 <b>Customer Support</b> – Detects sentiment in reviews & chat conversations to enhance customer experience.</li>
            <li>❤ <b>Mental Health Monitoring</b> – Identifies distress patterns, helping professionals offer timely support.</li>
            <li>📈 <b>Marketing & Brand Monitoring</b> – Deciphers customer emotions for product success & brand strategies.</li>
            <li>🌎 <b>Social Media & Content Analysis</b> – Tracks public opinion trends, predicting viral moments.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Business Objective
st.markdown('<h3 class="subheader">🎯 Business Objective</h3>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-box">
        <ul>
            <li>🧠 Understands emotions like joy, anger, sadness & more from text.</li>
            <li>🎯 Delivers high accuracy so it’s trustworthy & reliable.</li>
            <li>⚡ Runs efficiently in real-time for instant insights.</li>
            <li>📊 Provides clear & actionable insights for businesses.</li>
            <li>🗣 Think of it as an AI psychologist for text!</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Business Constraints
st.markdown('<h3 class="subheader">⚠ Business Constraints</h3>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-box">
        <ul>
            <li>⏳ <b>Speed vs. Accuracy</b> – It must be fast & precise.</li>
            <li>🔐 <b>Data Privacy & Compliance</b> – Emotions are personal; we ensure ethical handling.</li>
            <li>📡 <b>Scalability</b> – It should handle millions of texts efficiently.</li>
            <li>⚖ <b>Bias & Fairness</b> – The model must not misinterpret emotions based on gender, culture, or tone.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the Trained Model
model_path = "C:\\Users\\maren\\INNOMATICS_327\\ML_classes_327\\PROJECTS\\EMOTION_DETECTION_Project_7\\emotion_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        vectorizer, classifier = pickle.load(file)
else:
    st.error("🚨 Model file not found! Please check your working directory.")
    st.stop()

# Load Dataset for Performance Metrics
data_path = "C:\\Users\\maren\\INNOMATICS_327\\ML_classes_327\\PROJECTS\\EMOTION_DETECTION_Project_7\\combined_emotion.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    X = df["sentence"]
    y = df["emotion"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit vectorizer if not already fitted
    if not hasattr(vectorizer, 'vocabulary_'):
        vectorizer.fit(X_train)

    # Fit classifier if not already fitted
    if not hasattr(classifier, 'classes_'):
        X_train_transformed = vectorizer.transform(X_train)
        classifier.fit(X_train_transformed, y_train)

    # Transform text and predict
    X_test_transformed = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_transformed)

    # Compute performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.subheader("📊 Model Performance")
    st.write(f"✅ *Accuracy:* {accuracy:.2%}")
    st.write(f"✅ *F1-Score:* {f1:.2%}")
else:
    st.warning("⚠ Dataset not found! Performance metrics cannot be displayed.")

# User Input and Prediction
st.markdown('<h4 class="title">📝 Enter your text below:</h4>', unsafe_allow_html=True)
user_input = st.text_area("✍ Type your sentence here:")

if st.button("🔍 Detect Emotion"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = classifier.predict(transformed_input)[0]

        emotion_mapping = {
            "joy": ("😊", "#FFD700"),
            "sad": ("😢", "#6495ED"),
            "anger": ("😠", "#FF4500"),
            "fear": ("😨", "#A9A9A9"),
            "surprise": ("😲", "#32CD32"),
            "neutral": ("😐", "#808080"),
            "love": ("❤", "#FF1493"),
            "disgust": ("🤢", "#8B0000")
        }

        emoji, color = emotion_mapping.get(prediction.lower(), ("😶", "#000"))

        st.markdown(
            f'<p style="color:{color}; font-size:24px; font-weight:bold;">🎭 Predicted Emotion: {prediction} {emoji}</p>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠ Please enter text before predicting.")