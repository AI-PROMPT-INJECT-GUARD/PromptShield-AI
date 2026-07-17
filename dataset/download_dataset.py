from datasets import load_dataset
import pandas as pd
import os

# Create folders if they don't exist
os.makedirs("dataset/raw", exist_ok=True)

print("Downloading Prompt Injection Dataset...")

# Load dataset from Hugging Face
dataset = load_dataset(
    "neuralchemy/Prompt-injection-dataset",
    "core"
)

print("Download completed!")

# Convert dataset splits to DataFrames
train_df = pd.DataFrame(dataset["train"])
validation_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# Save CSV files
train_df.to_csv("dataset/raw/train.csv", index=False)
validation_df.to_csv("dataset/raw/validation.csv", index=False)
test_df.to_csv("dataset/raw/test.csv", index=False)

print("\nDataset saved successfully!")
print("Files created:")
print(" - dataset/raw/train.csv")
print(" - dataset/raw/validation.csv")
print(" - dataset/raw/test.csv")
