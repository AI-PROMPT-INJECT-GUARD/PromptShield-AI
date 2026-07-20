"""
inference.py
-------------
Loads the trained DistilBERT model (from Member 1's train_model.py,
saved into saved_model/) and exposes predict() for the FastAPI backend.

label 0 -> Safe Prompt
label 1 -> Prompt Injection
"""

import os
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_model"
)
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

LABELS = {0: "Safe", 1: "Prompt Injection"}

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"saved_model folder not found at '{os.path.abspath(MODEL_PATH)}'. "
            "Ask Member 1 for the saved_model folder (it's too large for GitHub) "
            "and place it in the project root, or set the MODEL_PATH env var."
        )

    _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.eval()
    return _tokenizer, _model


def predict(text: str) -> dict:
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item() * 100

    return {
        "label": LABELS[prediction],
        "is_injection": bool(prediction == 1),
        "confidence": round(confidence, 2),
    }