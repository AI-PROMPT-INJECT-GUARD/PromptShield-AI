import pandas as pd

# Load datasets
old_df = pd.read_csv("dataset/processed/train.csv")
safe_df = pd.read_csv("dataset/safe_prompts.csv")

# Keep only 2500 random safe prompts
safe_df = safe_df.sample(n=2500, random_state=42)

# Combine datasets
merged_df = pd.concat([old_df, safe_df], ignore_index=True)

# Shuffle dataset
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged dataset
merged_df.to_csv("dataset/processed/train_balanced.csv", index=False)

print("Dataset merged successfully!\n")
print(merged_df["label"].value_counts())
print("\nTotal samples:", len(merged_df))