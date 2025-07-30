import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import tempfile
from gtts import gTTS

# === Load model and labels ===
model = tf.keras.models.load_model("letter_model.h5")
with open("class_indices.json") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# === Streamlit Title ===
st.set_page_config(page_title="ASL Recognition", page_icon="🤟")
st.title("🤟 ASL Alphabet Recognition (Webcam)")

# === Define Capture function ===
def capture_and_predict():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Unable to access webcam.")
        return

    st.info("📸 Capturing frame... Hold hand steady!")
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("⚠️ Couldn't capture frame.")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    frame_cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    frame_resized = cv2.resize(frame_cropped, (64, 64))
    frame_resized = cv2.convertScaleAbs(frame_resized, alpha=1.3, beta=30)

    img = frame_resized / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_index = int(np.argmax(prediction))
    predicted_label = labels[predicted_index]
    confidence = float(np.max(prediction))

    # Save prediction into session state
    st.session_state['predicted_label'] = predicted_label.upper()

    st.image(frame_resized, caption="Captured Frame (Processed)", channels="RGB")
    st.markdown(f"### 🧠 Prediction: **{predicted_label.upper()}** ({confidence:.2f})")

# === UI Buttons ===
if st.button("📸 Capture & Predict"):
    capture_and_predict()

# If we have a prediction already stored
if 'predicted_label' in st.session_state:
    if st.button("🔊 Speak Prediction"):
        tts = gTTS(text=st.session_state['predicted_label'], lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format='audio/mp3')
            st.success("✅ Audio ready! Click ▶️ to hear it!")
