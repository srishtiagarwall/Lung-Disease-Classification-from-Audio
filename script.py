import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import os

# -------------------------
# 1. Page Configuration
# -------------------------
st.set_page_config(
    page_title="Lung Disease Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# 2. Load the Keras Model
# -------------------------
@st.cache_resource
def load_prediction_model(model_path="prediction_lung_disease_model.keras"):
    """
    Loads a Keras model from the specified path.
    Caches the result so it doesn't reload on every refresh.
    """
    model = load_model(model_path)
    return model

model = load_prediction_model()  # loads "prediction_lung_disease_model.keras" by default

# -------------------------
# 3. Audio Preprocessing Function
# -------------------------
def preprocess_audio(audio_file, sr_new=16000, duration=5, feature_type='mfcc'):
    """
    1) Loads audio at sr_new (e.g. 16 kHz).
    2) Pads or trims to a fixed 'duration' (e.g. 5 seconds).
    3) Extracts MFCC (or another feature like log-mel) and returns a shape suitable for model input.
    """
    # Load .wav
    x, sr = librosa.load(audio_file, sr=sr_new)

    # Ensure fixed length
    max_len = sr_new * duration
    if len(x) < max_len:
        x = np.pad(x, (0, max_len - len(x)))
    else:
        x = x[:max_len]

    # Feature extraction
    if feature_type == 'mfcc':
        feature = librosa.feature.mfcc(y=x, sr=sr_new)
    else:
        # Example: log-mel
        mel = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=128, fmax=8000)
        feature = librosa.power_to_db(mel, ref=np.max)

    # Reshape for CNN input: (rows, cols) -> (rows, cols, 1) -> (1, rows, cols, 1)
    feature = np.expand_dims(feature, axis=-1)
    feature = np.expand_dims(feature, axis=0)

    return feature

# -------------------------
# 4. Class Names
# -------------------------
# These should match the order from your training label encoder
CLASS_NAMES = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",
    "Healthy",
    "LRTI",
    "Pneumonia",
    "URTI"
]

# -------------------------
# 5. Streamlit UI
# -------------------------
st.title("Lung Disease Classification from Audio")
st.write("Upload a **.wav file** to predict the type of lung disease.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# If a file is uploaded...
if uploaded_file is not None:
    # Display an audio player
    st.audio(uploaded_file, format="audio/wav")

    # Save the uploaded file as a temp .wav for librosa to read
    temp_path = "temp_uploaded.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess audio
    input_features = preprocess_audio(temp_path, feature_type='mfcc')

    # Predict
    preds = model.predict(input_features)  # shape -> (1, n_classes)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds)) * 100
    predicted_class = CLASS_NAMES[class_idx]

    # Display results
    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
