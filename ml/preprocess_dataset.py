import os
import pandas as pd

# Folder paths
RAW_DATA_PATH = "dataset/raw"
PROCESSED_DATA_PATH = "dataset/processed"

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Files to preprocess
files = ["train.csv", "validation.csv", "test.csv"]

for file in files:

    print("=" * 60)
    print(f"Processing {file}")
    print("=" * 60)

    # Read CSV
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, file))

    print(f"\nOriginal Shape: {df.shape}")

    # Keep only required columns
    df = df[["text", "label"]]

    print(f"Shape after selecting columns: {df.shape}")

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Rows Found: {duplicate_count}")

    # Remove duplicates
    df = df.drop_duplicates()

    print(f"Shape after removing duplicates: {df.shape}")

    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Save processed dataset
    output_path = os.path.join(PROCESSED_DATA_PATH, file)
    df.to_csv(output_path, index=False)

    print(f"\nSaved cleaned dataset to: {output_path}")
    print("\n")