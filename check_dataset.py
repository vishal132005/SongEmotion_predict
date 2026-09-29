import os

# Path to our dataset
dataset_path = "archive"

# Count audio files
total_files = 0

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):
            total_files += 1

print("Total audio files:", total_files)