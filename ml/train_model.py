import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline

from build_dataset import build_dataset
from predictor import PromptInjectionPredictor


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "saved_model"
MODEL_PATH = MODEL_DIR / "promptshield_model.joblib"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / f"curated_{name}.csv")


def build_classifier() -> Pipeline:
    features = FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        C=4.0,
        class_weight="balanced",
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    candidates = np.arange(0.50, 0.991, 0.01)
    best = None
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        candidate = {
            "threshold": float(round(threshold, 2)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        if best is None:
            best = candidate
            continue
        current_key = (
            candidate["precision"] >= 0.98,
            candidate["false_positive_rate"] <= 0.02,
            candidate["precision"],
            candidate["recall"],
            candidate["f1"],
            candidate["threshold"],
        )
        best_key = (
            best["precision"] >= 0.98,
            best["false_positive_rate"] <= 0.02,
            best["precision"],
            best["recall"],
            best["f1"],
            best["threshold"],
        )
        if current_key > best_key:
            best = candidate
    return best


def evaluate_split(name: str, labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    pr_auc = average_precision_score(labels, scores)
    return {
        "split": name,
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_benchmark(model_dir: Path) -> dict:
    benchmark_df = load_split("benchmark")
    predictor = PromptInjectionPredictor(model_dir=model_dir)
    results = []
    for text, label in zip(benchmark_df["text"].astype(str), benchmark_df["label"].astype(int)):
        prediction = predictor.predict_text(text)
        predicted_label = 1 if prediction["label"] == "PROMPT_INJECTION" else 0
        results.append(
            {
                "text": text,
                "expected_label": int(label),
                "predicted_label": predicted_label,
                "injection_score": float(prediction["injection_score"]),
                "decision_source": prediction["decision_source"],
                "matched_rule": prediction["matched_rule"],
            }
        )

    result_df = pd.DataFrame(results)
    labels = result_df["expected_label"].to_numpy()
    predictions = result_df["predicted_label"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    safe_false_positives = result_df[(result_df["expected_label"] == 0) & (result_df["predicted_label"] == 1)]["text"].tolist()
    false_negatives = result_df[(result_df["expected_label"] == 1) & (result_df["predicted_label"] == 0)]["text"].tolist()
    result_df.to_csv(DATASET_DIR / "curated_benchmark_predictions.csv", index=False)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "safe_false_positives": safe_false_positives,
        "false_negatives": false_negatives,
    }


def main() -> None:
    stats = build_dataset()
    print("Curated dataset built:")
    print(json.dumps(stats, indent=2))

    train_df = load_split("train")
    validation_df = load_split("validation")
    test_df = load_split("test")

    pipeline = build_classifier()
    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(int))

    validation_scores = pipeline.predict_proba(validation_df["text"].astype(str))[:, 1]
    threshold_info = choose_threshold(validation_df["label"].to_numpy(), validation_scores)
    threshold = threshold_info["threshold"]

    test_scores = pipeline.predict_proba(test_df["text"].astype(str))[:, 1]
    validation_report = evaluate_split("validation", validation_df["label"].to_numpy(), validation_scores, threshold)
    test_report = evaluate_split("test", test_df["label"].to_numpy(), test_scores, threshold)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_type": "tfidf_logistic_regression",
        "decision_threshold": threshold,
        "threshold_selection": threshold_info,
        "model_path": str(MODEL_PATH),
        "dataset_stats_path": str(DATASET_DIR / "curated_dataset_stats.json"),
    }
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    benchmark_report = evaluate_benchmark(MODEL_DIR)
    report = {
        "validation": validation_report,
        "test": test_report,
        "benchmark": benchmark_report,
    }
    with (MODEL_DIR / "evaluation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Threshold selection:")
    print(json.dumps(threshold_info, indent=2))
    print("Evaluation report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
