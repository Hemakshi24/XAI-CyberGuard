
from pathlib import Path
import pandas as pd
import numpy as np

# Input aur output folders
data_folder = Path("data/cicids2017")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "cicids2017_sample.csv"

csv_files = sorted(data_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {data_folder.resolve()}"
    )

# Sampling settings: har class ke liye maximum records
CHUNK_SIZE = 50000
SAMPLE_PER_LABEL_PER_CHUNK = 2000
MAX_PER_LABEL = 20000

sampled_chunks = []

for file_path in csv_files:
    print(f"\nProcessing: {file_path.name}")

    try:
        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            # Column names ke extra spaces hatao
            chunk.columns = chunk.columns.str.strip()

            # Label column identify karo
            label_columns = [
                col for col in chunk.columns
                if col.lower() == "label"
            ]

            if not label_columns:
                print("Skipping: Label column not found.")
                continue

            label_col = label_columns[0]

            # Labels clean karo
            chunk[label_col] = (
                chunk[label_col]
                .astype("string")
                .str.strip()
                .str.replace("\ufffd", "-", regex=False)
            )

            chunk = chunk.dropna(subset=[label_col])

            # Feature columns ko numeric banao
            feature_cols = [
                col for col in chunk.columns
                if col != label_col
            ]

            for col in feature_cols:
                chunk[col] = pd.to_numeric(
                    chunk[col], errors="coerce"
                )

            # Infinite values ko missing values banao
            chunk[feature_cols] = chunk[feature_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            # Har chunk se har label ke random records lo
            for label, group in chunk.groupby(label_col):
                take = min(
                    SAMPLE_PER_LABEL_PER_CHUNK,
                    len(group)
                )

                sampled_chunks.append(
                    group.sample(n=take, random_state=42)
                )

        print("File processed.")

    except Exception as error:
        print(f"Error processing {file_path.name}: {error}")
        raise

if not sampled_chunks:
    raise ValueError("No labelled records were collected.")

# Samples combine karo
data = pd.concat(sampled_chunks, ignore_index=True)

label_col = next(
    col for col in data.columns
    if col.lower() == "label"
)

# Har class ke records ki maximum limit rakho
#data = (
 #   data.groupby(label_col, group_keys=False)
#    .apply(
 #       lambda group: group.sample(
  #          n=min(len(group), MAX_PER_LABEL),
   #         random_state=42
    #    ),
   #     include_groups=True
  #  )
 #   .reset_index(drop=True)
#)


sampled_groups = []

for label, group in data.groupby(label_col):
    take = min(len(group), MAX_PER_LABEL)

    sampled_group = group.sample(
        n=take,
        random_state=42
    )

    sampled_groups.append(sampled_group)

data = pd.concat(sampled_groups, ignore_index=True)


# Jinke saare values missing hain, aise features hatao
feature_cols = [
    col for col in data.columns
    if col != label_col
]

all_missing = data[feature_cols].isna().all()
columns_to_drop = all_missing[all_missing].index.tolist()

data = data.drop(columns=columns_to_drop)

# Processed dataset save karo
data.to_csv(output_file, index=False)

print("\nPreprocessing complete!")
print("Output file:", output_file)
print("Dataset shape:", data.shape)
print("\nRecords per label:")
print(data[label_col].value_counts().to_string())
print("\nFeatures removed because all values were missing:")
print(columns_to_drop)
