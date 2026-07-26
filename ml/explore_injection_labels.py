import pandas as pd

df = pd.read_csv("dataset/processed/train.csv")

print("Label Counts:")
print(df["label"].value_counts())

print("\nFirst 20 Prompt Injection Examples:\n")
print(df[df["label"] == 1][["text"]].head(20).to_string(index=False))