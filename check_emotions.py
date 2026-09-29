import os
from collections import Counter

dataset_path = "archive"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

emotion_counts = Counter()
total = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.endswith(".wav"):
            parts = file.split("-")

            if len(parts) >= 3:
                emotion_code = parts[2]

                emotion = emotion_map.get(
                    emotion_code, "unknown"
                )

                emotion_counts[emotion] += 1
                total += 1

print("Total audio files:", total)

print("\nEmotion Distribution:")

for emotion, count in sorted(emotion_counts.items()):
    print(emotion, ":", count)