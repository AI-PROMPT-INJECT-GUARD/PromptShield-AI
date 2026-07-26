from datasets import load_dataset

print("Downloading Alpaca dataset...")

dataset = load_dataset("yahma/alpaca-cleaned")

print(dataset)

print("\nFirst Sample:")
print(dataset["train"][0])