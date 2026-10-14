
import os
import joblib
import librosa
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load("emotion_model.pkl")

# Ask the user for an audio file path
audio_path = input("Enter the path of your audio file: ").strip().strip('"')

# Check whether the file exists
if not os.path.isfile(audio_path):
    print("Error: Audio file not found!")
    raise SystemExit

# Load audio with the same settings used during training
audio, sr = librosa.load(
    audio_path,
    sr=22050,
    mono=True
)

# Extract 13 MFCC coefficients
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

# Calculate mean and standard deviation
mfcc_mean = np.mean(mfcc, axis=1)
mfcc_std = np.std(mfcc, axis=1)

# Combine into 26 features
features = np.concatenate([
    mfcc_mean,
    mfcc_std
])

# Use the same feature names and order as the training dataset
feature_names = pd.read_csv("features.csv", nrows=0).columns.drop("emotion")
input_data = pd.DataFrame([features], columns=feature_names)

# Predict emotion
prediction = model.predict(input_data)[0]

print("\n------------------------------")
print("Predicted Emotion:", prediction.upper())
print("------------------------------")

# Display model probabilities, if available
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_data)[0]

    print("\nPrediction probabilities:")
    for emotion, probability in zip(model.classes_, probabilities):
        print(f"{emotion.capitalize()}: {probability * 100:.2f}%")
