from datasets import load_dataset
import pandas as pd
import os

print("Downloading Prompt Injection Dataset...")

dataset = load_dataset("Shomi28/prompt-injection-dataset")

os.makedirs("dataset/shomi_dataset", exist_ok=True)

# Save all splits
pd.DataFrame(dataset["train"]).to_csv(
    "dataset/shomi_dataset/train.csv",
    index=False
)

pd.DataFrame(dataset["validation"]).to_csv(
    "dataset/shomi_dataset/validation.csv",
    index=False
)

pd.DataFrame(dataset["test"]).to_csv(
    "dataset/shomi_dataset/test.csv",
    index=False
)

print("✅ All datasets saved successfully!")
print(dataset)