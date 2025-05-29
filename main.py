import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load label encoder (manually set it based on your training order)
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
    'siren', 'street_music'
]
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)

# Feature extraction
def extract_mfcc(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, sr=None)
    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(-1, 40, 1)  # reshape for model

# Paths
INPUT_DIR = "input/input/100648-1-0-0.wav"
MODEL_PATH = "models/model_fold10.keras"

# Load model
model = load_model(MODEL_PATH)

# Scan input folder
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".wav"):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"\n Processing: {filename}")

        try:
            features = extract_mfcc(file_path)
            prediction = model.predict(features)
            predicted_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

            print(f" Prediction: {predicted_label}")

        except Exception as e:
            print(f" Error processing {filename}: {e}")