
from pathlib import Path
import pandas as pd

data_folder = Path("data/cicids2017")
csv_files = sorted(data_folder.rglob("*.csv"))

if not csv_files:
    print("No CSV files found. Check the dataset folder.")
else:
    for file_path in csv_files:
        print("\n" + "=" * 60)
        print("File:", file_path.name)

        try:
            # Sirf Label column read karenge, poori CSV nahi
            df = pd.read_csv(
                file_path,
                usecols=lambda col: col.strip().lower() == "label",
                low_memory=False
            )

            if df.empty or len(df.columns) == 0:
                print("Label column not found.")
                continue

            label_column = df.columns[0]
            labels = df[label_column].astype("string").str.strip()

            print("Total records:", len(labels))
            print("\nRecords per label:")
            print(labels.value_counts(dropna=False).to_string())

        except Exception as error:
            print("Error reading file:", error)
