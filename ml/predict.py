import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")

# Load trained model
model = DistilBertForSequenceClassification.from_pretrained("./saved_model")

# Put model in evaluation mode
model.eval()

print("Model loaded successfully!")

# Take input from the user
text = input("\nEnter a prompt: ")
# Tokenize the input
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)
# Disable gradient calculation
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probabilities = F.softmax(outputs.logits, dim=1)

# Get predicted class
prediction = torch.argmax(probabilities, dim=1).item()

# Confidence of predicted class
confidence = probabilities[0][prediction].item() * 100
# Display result
if prediction == 0:
    print("\nPrediction: ✅ Safe Prompt")
else:
    print("\nPrediction: 🚨 Prompt Injection")

print(f"Confidence: {confidence:.2f}%")
