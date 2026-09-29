import os
from collections import Counter

dataset_path = "archive"

selected_emotions = {
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

counts = Counter()

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if not file.endswith(".wav"):
            continue

        parts = file.split("-")

        if len(parts) == 7:
            emotion_code = parts[2]

            if emotion_code in selected_emotions:
                counts[selected_emotions[emotion_code]] += 1

print("Selected emotion counts:")

for emotion, count in sorted(counts.items()):
    print(emotion, ":", count)

print("Total selected:", sum(counts.values()))
