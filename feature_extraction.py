
import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "archive"

emotion_map = {
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

features = []

print("Extracting MFCC features...")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.endswith(".wav"):
            continue

        parts = file.split("-")

        if len(parts) != 7:
            continue

        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        emotion = emotion_map[emotion_code]

        file_path = os.path.join(root, file)

        try:
            audio, sr = librosa.load(
                file_path,
                sr=22050,
                mono=True
            )

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13
            )

            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            feature_vector = np.concatenate(
                (mfcc_mean, mfcc_std)
            )

            row = feature_vector.tolist()
            row.append(emotion)

            features.append(row)

        except Exception as e:
            print("Error processing:", file, e)

columns = []

for i in range(1, 14):
    columns.append(f"mfcc_mean_{i}")

for i in range(1, 14):
    columns.append(f"mfcc_std_{i}")

columns.append("emotion")

df = pd.DataFrame(features, columns=columns)

df.to_csv("features.csv", index=False)

print("\nFeature extraction completed!")
print("Total samples:", len(df))
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())