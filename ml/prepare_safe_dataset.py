import pandas as pd
from datasets import load_dataset

print("Loading Alpaca dataset...")

dataset = load_dataset("yahma/alpaca-cleaned")

rows = []

for sample in dataset["train"]:
    instruction = sample["instruction"].strip()
    input_text = sample["input"].strip()

    if input_text:
        text = instruction + "\n" + input_text
    else:
        text = instruction

    rows.append({
        "text": text,
        "label": 0
    })

safe_df = pd.DataFrame(rows)

print(safe_df.head())
print("\nTotal Safe Prompts:", len(safe_df))

safe_df.to_csv("dataset/safe_prompts.csv", index=False)

print("\nSaved successfully!")