import pandas as pd

df = pd.read_csv("dataset/shomi_dataset/train.csv")

print("="*60)
print("Dataset Shape")
print("="*60)
print(df.shape)

print("\n")

print("="*60)
print("Label Distribution")
print("="*60)
print(df["label_name"].value_counts())

print("\n")

print("="*60)
print("First 10 Safe Prompts")
print("="*60)
print(df[df["label"]==0][["text"]].head(10).to_string(index=False))

print("\n")

print("="*60)
print("First 10 Injection Prompts")
print("="*60)
print(df[df["label"]==1][["text"]].head(10).to_string(index=False))