import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Custom Styling
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

# Display an Image
image_path = "Innomatics-Logo1.png"
if os.path.exists(image_path):
    st.image(image_path, width=700)
else:
    st.warning("Image not found! Check the file path.")

# Streamlit UI
st.title("😊 Emotion Detection from Text")

# 📍 a) Business Problem: Emotion Detection Using AI 🤖🎭
st.subheader("📍 a) Business Problem: Emotion Detection Using AI 🤖🎭")

st.markdown(
    """
    <div style="background-color:#f9f9f9; padding:10px; border-radius:10px;">
        <p style="font-size:18px; font-weight:bold; color:#ff5733;">
            In an era driven by <b>data & emotions</b>, understanding human sentiments from text is game-changing! 🚀
            Imagine an AI that "feels" what you're saying—here's how it helps:
        </p>
        <p style="font-size:16px;">🔹 <b>🤝 Customer Support</b> – Detects sentiment in reviews & chat conversations to enhance customer experience.</p>
        <p style="font-size:16px;">🔹 <b>🧠 Mental Health Monitoring</b> – Identifies distress patterns, helping professionals offer timely support.</p>
        <p style="font-size:16px;">🔹 <b>📢 Marketing & Brand Monitoring</b> – Deciphers customer emotions for product success & brand strategies.</p>
        <p style="font-size:16px;">🔹 <b>📊 Social Media & Content Analysis</b> – Tracks public opinion trends, predicting viral moments!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Business Objective
st.subheader("📍 b) Business Objective 🎯")
st.write("""
We aim to craft an AI-powered **emotion detection model** that:

✅ **Understands** emotions like joy, anger, sadness & more from text!  
✅ **Delivers High Accuracy** – so it’s **trustworthy & reliable**.  
✅ **Runs Efficiently in Real-time** – for instant insights! ⏳  
✅ **Is Business-Friendly** – providing clear & actionable insights!  

💡 Think of it as an **AI psychologist for text!** 🧐📜
""")

# Business Constraints
st.subheader("📍 c) Business Constraints ⚖️")
st.write("""
To build a **rock-solid model**, we must tackle:

🚀 **Speed vs. Accuracy** – It must be **fast & precise**!  
🔐 **Data Privacy & Compliance** – Emotions are **personal**; we ensure **ethical handling**.  
📈 **Scalability** – It should handle **millions of texts** without breaking a sweat! 💪  
⚖️ **Bias & Fairness** – A fair AI that doesn't misinterpret emotions based on **gender, culture, or tone**.  
""")

# Data Understanding
st.subheader("📍 d) Data Understanding 📊")
st.write("""
Our AI learns from **real-world emotions** using **massive datasets!**  

🗂️ **Dataset Size:** **422,746 text samples** – A rich emotional spectrum! 🌈  
📌 **Key Columns:**  
   📝 **sentence** – The text we analyze (e.g., _"I'm thrilled about my promotion!"_ 🥳)  
   🎭 **emotion** – The AI’s prediction (e.g., **"Joy"** 😊, **"Sadness"** 😢, **"Anger"** 😠)  

💡 **More data = Smarter AI**! 🤖📚
""")

# Load the trained model
model_path = "emotion_detection_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Ensure the model is correctly loaded
    if hasattr(model, "named_steps"):
        vectorizer = model.named_steps.get("vectorizer")
        classifier = model.named_steps.get("classifier")
    else:
        st.error("❌ Invalid model format. Ensure it's a valid scikit-learn pipeline.")
        st.stop()
else:
    st.error("❌ Model file not found! Please check your working directory.")
    st.stop()

# Load dataset for performance metrics
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

# Emotion Mapping with Colors
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

# User Input
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