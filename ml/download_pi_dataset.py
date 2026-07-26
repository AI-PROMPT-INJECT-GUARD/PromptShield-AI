from datasets import load_dataset
import pandas as pd
import os

print("Downloading Prompt Injection dataset...")

# Load dataset from Hugging Face
dataset = load_dataset("deepset/prompt-injections")

print(dataset)

os.makedirs("dataset/pi_dataset", exist_ok=True)

# Save train split
train_df = pd.DataFrame(dataset["train"])
train_df.to_csv("dataset/pi_dataset/train.csv", index=False)

print("\nDataset downloaded successfully!")
print("Total samples:", len(train_df))
print("\nColumns:")
print(train_df.columns.tolist())

print("\nFirst 5 rows:")
print(train_df.head())
