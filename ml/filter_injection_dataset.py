import pandas as pd

df = pd.read_csv("dataset/pi_dataset/train.csv")

safe_df = df[df["label"] == 0]

patterns = [
    "ignore",
    "forget",
    "system prompt",
    "prompt text",
    "reveal",
    "override",
    "developer mode",
    "dan",
    "jailbreak",
    "hidden prompt",
    "ignore the above",
    "ignore previous",
    "disregard",
    "previous instructions",
    "instructions"
]

inj_df = df[df["label"] == 1].copy()

inj_df = inj_df[
    inj_df["text"].str.lower().apply(
        lambda x: any(p in x for p in patterns)
    )
]

print("Safe:", len(safe_df))
print("Injection:", len(inj_df))

inj_df.to_csv(
    "dataset/pi_dataset/final_prompt_injection.csv",
    index=False
)

print("Saved!")