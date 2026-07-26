import argparse
import json

from predictor import PromptInjectionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="PromptShield AI predictor")
    parser.add_argument("--text", type=str, help="Prompt text to classify")
    parser.add_argument("--model-dir", type=str, default="saved_model", help="Directory containing the saved model")
    args = parser.parse_args()

    predictor = PromptInjectionPredictor(model_dir=args.model_dir)

    if args.text:
        result = predictor.predict_text(args.text)
        print(json.dumps(result, indent=2))
        return

    print("Model loaded successfully.")
    while True:
        try:
            text = input("\nEnter a prompt (or 'exit' to quit): ").strip()
        except EOFError:
            break
        if not text or text.lower() == "exit":
            break
        result = predictor.predict_text(text)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
