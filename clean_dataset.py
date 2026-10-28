
from pathlib import Path
import pandas as pd
import numpy as np

input_file = Path("data/processed/cicids2017_sample.csv")
output_file = Path("data/processed/cicids2017_clean.csv")

# Dataset load karo
data = pd.read_csv(input_file)

print("Original shape:", data.shape)

# Infinite values ko missing values mein convert karo
data = data.replace([np.inf, -np.inf], np.nan)

# Duplicate rows remove karo
before_duplicates = len(data)
data = data.drop_duplicates()
removed_duplicates = before_duplicates - len(data)

# Label missing ho to row remove karo
data = data.dropna(subset=["Label"])

# Features ko numeric values mein rakho
feature_cols = [col for col in data.columns if col != "Label"]

for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Feature missing values ko column median se fill karo
for col in feature_cols:
    median_value = data[col].median()

    if pd.notna(median_value):
        data[col] = data[col].fillna(median_value)
    else:
        data[col] = data[col].fillna(0)

# Clean dataset save karo
data.to_csv(output_file, index=False)

print("\nCleaning complete!")
print("Cleaned shape:", data.shape)
print("Duplicate rows removed:", removed_duplicates)
print("Missing values remaining:", data.isna().sum().sum())
print("Saved to:", output_file)

print("\nUpdated class distribution:")
print(data["Label"].value_counts().to_string())
