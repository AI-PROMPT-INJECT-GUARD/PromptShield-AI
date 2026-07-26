import pandas as pd

df = pd.read_csv("dataset/pi_dataset/train.csv")

print("=" * 50)
print("Label Distribution")
print("=" * 50)
print(df["label"].value_counts())

print("\n")

print("=" * 50)
print("SAFE PROMPTS")
print("=" * 50)
print(df[df["label"] == 0].head(10).to_string(index=False))

print("\n")

print("=" * 50)
print("PROMPT INJECTIONS")
print("=" * 50)
print(df[df["label"] == 1].head(10).to_string(index=False))
