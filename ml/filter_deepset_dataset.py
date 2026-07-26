import pandas as pd

# Load dataset
df = pd.read_csv("dataset/pi_dataset/train.csv")

# Keep all safe prompts
safe_df = df[df["label"] == 0]

# Words that usually indicate roleplay rather than prompt injection
remove_keywords = [
    "act as",
    "pretend",
    "roleplay",
    "role-play",
    "style of",
    "you are xi",
    "you are a",
    "you are an",
    "debater",
    "essay",
    "opinion piece"
]

def keep_prompt(text):
    text = text.lower()

    # Remove role-play prompts
    for keyword in remove_keywords:
        if keyword in text:
            return False

    # Keep genuine prompt injection patterns
    injection_keywords = [
        "ignore previous",
        "forget previous",
        "forget all previous",
        "system prompt",
        "prompt text",
        "instructions",
        "override",
        "developer mode",
        "ignore the above",
        "stop, ignore"
    ]

    for keyword in injection_keywords:
        if keyword in text:
            return True

    return False


inj_df = df[
    (df["label"] == 1) &
    (df["text"].apply(keep_prompt))
]

print("=" * 50)
print("Safe Prompts :", len(safe_df))
print("Prompt Injections :", len(inj_df))
print("=" * 50)

# Save cleaned injection dataset
inj_df.to_csv(
    "dataset/pi_dataset/clean_prompt_injection.csv",
    index=False
)

print("\nSaved as dataset/pi_dataset/clean_prompt_injection.csv")