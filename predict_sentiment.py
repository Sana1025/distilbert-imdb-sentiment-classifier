#!/usr/bin/env python3
"""Predict IMDb sentiment with a trained DistilBERT checkpoint."""

import argparse
import json
from pathlib import Path

from project_config import PROJECT_DIR

DEFAULT_MODEL_DIR = PROJECT_DIR / "runs" / "seed_42" / "best_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify movie reviews with a trained DistilBERT model"
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Review to classify. Repeat --text to classify multiple reviews.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Saved model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum number of tokens (default: 256)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    return parser.parse_args()


def select_device(torch, requested_device):
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_name(model, label_id):
    mapping = model.config.id2label
    return str(mapping.get(label_id, mapping.get(str(label_id), label_id))).upper()


def classify_review(text, tokenizer, model, device, max_length, torch):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1).cpu()

    predicted_id = int(torch.argmax(probabilities).item())
    scores = [float(score) for score in probabilities]
    return predicted_id, scores, len(encoded["input_ids"][0])


def print_prediction(text, tokenizer, model, device, max_length, torch):
    predicted_id, scores, token_count = classify_review(
        text, tokenizer, model, device, max_length, torch
    )

    print("\nReview:")
    print(text)
    print(f"\nPrediction: {label_name(model, predicted_id)}")
    print(f"Confidence: {100.0 * scores[predicted_id]:.2f}%")
    print(f"Tokens used: {token_count}/{max_length}")
    print("Probabilities:")
    for label_id, score in enumerate(scores):
        print(f"  {label_name(model, label_id):8s}: {100.0 * score:.2f}%")


def print_saved_metrics(model_dir):
    metrics_path = model_dir.parent / "metrics.json"
    if not metrics_path.is_file():
        return

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"Saved test accuracy: {100.0 * metrics['test_accuracy']:.2f}%")
        print(f"Saved test F1: {100.0 * metrics['test_f1']:.2f}%")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        print(f"Note: could not read saved metrics from {metrics_path}")


def main():
    args = parse_args()

    if args.max_length < 1:
        raise ValueError("--max-length must be a positive integer.")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"Saved model not found: {args.model_dir}\n"
            "Wait for the training job to complete, or provide --model-dir."
        )
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"The directory does not contain a complete model: {args.model_dir}"
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = select_device(torch, args.device)
    print(f"Loading model: {args.model_dir}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_dir), local_files_only=True
    )
    model.to(device)
    model.eval()

    print_saved_metrics(args.model_dir)

    if args.text:
        for review in args.text:
            if review.strip():
                print_prediction(
                    review.strip(),
                    tokenizer,
                    model,
                    device,
                    args.max_length,
                    torch,
                )
        return

    print("\nInteractive mode")
    print("Type a movie review and press Enter.")
    print("Type quit to finish.")

    while True:
        try:
            review = input("\nReview> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFinished.")
            break

        if review.lower() in {"quit", "exit"}:
            print("Finished.")
            break
        if not review:
            print("Please enter a review, or type quit.")
            continue

        print_prediction(
            review, tokenizer, model, device, args.max_length, torch
        )


if __name__ == "__main__":
    main()
