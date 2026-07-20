"""
attack_categories.py
---------------------
The trained model only outputs Safe / Injection (binary). This module adds
the rule-based layer on top: it figures out WHICH KIND of injection was
detected, explains why, and produces a sanitized "safe prompt" version.

This is only run when the model already flagged a prompt as Prompt Injection.
"""

import re

CATEGORY_PATTERNS = [
    ("Instruction Override", [
        r"ignore (?:all |any |previous |the )*instructions",
        r"disregard (?:all |previous |your )*instructions",
        r"forget (?:all |your )*(?:instructions|rules)",
        r"new instructions?:",
    ]),
    ("Jailbreak / Persona Hijack", [
        r"\bdan\b",
        r"do anything now",
        r"you are now .*(without|no) restrictions",
        r"pretend you (are|have) no (rules|restrictions|filters)",
        r"act as .* with no (limitations|restrictions)",
        r"jailbreak",
    ]),
    ("System Prompt Leakage", [
        r"(show|reveal|print|repeat|output) (me )?(your|the) (system prompt|instructions|rules)",
        r"what (are|were) your (initial|original) instructions",
    ]),
    ("Role-Play Manipulation", [
        r"role[- ]?play as",
        r"pretend (to be|you are)",
        r"you are now playing the role",
        r"from now on you are",
    ]),
    ("Encoding / Obfuscation", [
        r"base64",
        r"rot13",
        r"decode the following",
        r"\\x[0-9a-fA-F]{2}",
    ]),
    ("Data Exfiltration", [
        r"(api key|password|secret|credentials|token)",
        r"send (this|it) to",
        r"exfiltrate",
    ]),
    ("Delimiter / Prompt Injection Marker", [
        r"</?(system|assistant|user)>",
        r"---\s*end of prompt",
        r"\[system\]",
    ]),
]

EXPLANATIONS = {
    "Instruction Override": "The prompt tries to make the model ignore or discard its original instructions.",
    "Jailbreak / Persona Hijack": "The prompt tries to assign the model an unrestricted persona to bypass safety rules.",
    "System Prompt Leakage": "The prompt tries to extract the model's hidden system instructions.",
    "Role-Play Manipulation": "The prompt uses a role-play framing to push the model outside its normal behavior.",
    "Encoding / Obfuscation": "The prompt hides instructions using encoding to evade detection filters.",
    "Data Exfiltration": "The prompt attempts to extract sensitive data such as keys or credentials.",
    "Delimiter / Prompt Injection Marker": "The prompt injects fake system/user markers to confuse the model about who is speaking.",
    "General Prompt Injection": "Injection-like patterns were detected that don't match a specific known category.",
}


def categorize(text: str) -> str:
    lowered = text.lower()
    for category, patterns in CATEGORY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return category
    return "General Prompt Injection"


def explain(category: str, confidence: float) -> str:
    base = EXPLANATIONS.get(category, EXPLANATIONS["General Prompt Injection"])
    return f"{base} (Model confidence: {confidence:.2f}%)"


def safe_rewrite(text: str, category: str) -> str:
    sanitized = text
    for _, patterns in CATEGORY_PATTERNS:
        for pattern in patterns:
            sanitized = re.sub(pattern, "[filtered]", sanitized, flags=re.IGNORECASE)

    if sanitized.strip() == text.strip():
        return (
            "This prompt was flagged as unsafe and could not be automatically "
            "sanitized. Please rephrase without instructions that try to "
            "override system behavior."
        )
    return sanitized.strip()