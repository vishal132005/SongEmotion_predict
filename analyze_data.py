import pandas as pd

# Load the extracted dataset
df = pd.read_csv("features.csv")

# 1. Display dataset information
print("Dataset Information:")
print(df.info())

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 3. Count samples in each emotion
print("\nEmotion Distribution:")
print(df["emotion"].value_counts())

# 4. Display statistical summary
print("\nStatistical Summary:")
print(df.describe())