
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/cicids2017_sample.csv")

data = pd.read_csv(file_path)

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:", data.isna().sum().sum())
print("Duplicate rows:", data.duplicated().sum())

print("\nClass distribution:")
print(data["Label"].value_counts())
