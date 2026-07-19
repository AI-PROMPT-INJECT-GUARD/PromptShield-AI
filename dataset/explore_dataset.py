import pandas as pd
import os

# Path to the raw dataset folder
DATA_PATH = "dataset/raw"

# Files to explore
files = ["train.csv", "validation.csv", "test.csv"]

for file in files:
    print("=" * 60)
    print(f"Exploring {file}")
    print("=" * 60)

    # Read the CSV
    df = pd.read_csv(os.path.join(DATA_PATH, file))

    # Shape
    print("\nShape:")
    print(df.shape)

    # Column names
    print("\nColumns:")
    print(df.columns.tolist())

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # First five rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Label distribution
    print("\nLabel Distribution:")
    print(df["label"].value_counts())

    # Label percentage
    print("\nLabel Percentage:")
    print(df["label"].value_counts(normalize=True) * 100)
    print("\n")
