import json
import re
from pathlib import Path

import joblib


RULES = [
    ("override_previous_instructions", re.compile(r"\b(ignore|disregard|forget|override|dismiss)\b.{0,60}\b(previous|prior|above|earlier)\b.{0,60}\b(instruction|instructions|rule|rules|prompt|context)\b", re.IGNORECASE)),
    ("reveal_system_prompt", re.compile(r"\b(reveal|show|print|repeat|output|display|dump|summarize)\b.{0,60}\b(system prompt|hidden instructions?|internal instructions?|developer instructions?|secret rules?|initialization prompt|configuration)\b", re.IGNORECASE)),
    ("bypass_safety", re.compile(r"\b(bypass|disable|suspend|remove)\b.{0,40}\b(safety|guardrails?|restrictions?|filters?|protections?)\b", re.IGNORECASE)),
    ("dan_persona", re.compile(r"\b(act as dan|dan mode|developer mode)\b", re.IGNORECASE)),
    ("agent_override", re.compile(r"\b(skip|bypass)\b.{0,40}\b(validation|check_safety|safety check)\b", re.IGNORECASE)),
    ("next_agent_override", re.compile(r"\bsend to next agent\b.{0,40}\boverride safety", re.IGNORECASE)),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


class PromptInjectionPredictor:
    def __init__(self, model_dir: str | Path = "saved_model") -> None:
        self.model_dir = Path(model_dir)
        self.model = None
        self.threshold = 0.90

        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.threshold = float(metadata.get("decision_threshold", self.threshold))
            model_path = metadata.get("model_path")
            if model_path:
                self.model = joblib.load(model_path)

    def match_rule(self, text: str) -> str | None:
        for name, pattern in RULES:
            if pattern.search(text):
                return name
        return None

    def predict_text(self, text: str) -> dict:
        cleaned = normalize_text(text)
        if not cleaned:
            return {
                "label": "SAFE",
                "injection_score": 0.0,
                "decision_source": "empty_input",
                "matched_rule": None,
            }

        matched_rule = self.match_rule(cleaned)
        if matched_rule is not None:
            return {
                "label": "PROMPT_INJECTION",
                "injection_score": 1.0,
                "decision_source": "rule",
                "matched_rule": matched_rule,
            }

        if self.model is None:
            raise FileNotFoundError(f"No trained sklearn model found in {self.model_dir}")

        injection_score = float(self.model.predict_proba([cleaned])[0][1])
        label = "PROMPT_INJECTION" if injection_score >= self.threshold else "SAFE"
        return {
            "label": label,
            "injection_score": injection_score,
            "decision_source": "model",
            "matched_rule": None,
        }


def predict_text(text: str, model_dir: str | Path = "saved_model") -> dict:
    predictor = PromptInjectionPredictor(model_dir=model_dir)
    return predictor.predict_text(text)
