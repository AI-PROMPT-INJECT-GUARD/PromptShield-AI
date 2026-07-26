import pandas as pd

df = pd.read_csv("dataset/pi_dataset/train.csv")

positive_df = df[df["label"] == 1].copy()

positive_df.insert(0, "keep", "")

positive_df.to_csv(
    "dataset/pi_dataset/review_prompt_injections.csv",
    index=False
)

print("Exported", len(positive_df), "examples.")
print("Saved to dataset/pi_dataset/review_prompt_injections.csv")