"""
inference.py
-------------
Loads the merged prompt injection predictor from the repository root and
normalizes its output to the FastAPI backend response contract.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predictor import PromptInjectionPredictor

DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_model"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

LABELS = {
    "SAFE": "Safe",
    "PROMPT_INJECTION": "Prompt Injection",
}

_predictor = None


def load_model() -> PromptInjectionPredictor:
    global _predictor

    if _predictor is not None:
        return _predictor

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"saved_model folder not found at '{MODEL_PATH.resolve()}'. "
            "Place the trained model assets in the project root or set the MODEL_PATH env var."
        )

    _predictor = PromptInjectionPredictor(model_dir=MODEL_PATH)
    return _predictor


def predict(text: str) -> dict:
    predictor = load_model()
    result = predictor.predict_text(text)

    label = LABELS.get(result["label"], result["label"])
    injection_score = float(result.get("injection_score", 0.0))
    is_injection = result["label"] == "PROMPT_INJECTION"
    confidence = injection_score * 100 if is_injection else (1.0 - injection_score) * 100

    return {
        "label": label,
        "is_injection": is_injection,
        "confidence": round(confidence, 2),
    }
