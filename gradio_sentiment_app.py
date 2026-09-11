#!/usr/bin/env python3
"""Private Gradio demo for the fine-tuned DistilBERT IMDb classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from project_config import PROJECT_DIR

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = PROJECT_DIR / "runs" / "seed_42" / "best_model"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


class SentimentPredictor:
    """Load the saved model once and reuse it for every web request."""

    def __init__(self, model_dir: Path, device: str, max_length: int) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available.")

        self.device = torch.device(device)
        self.max_length = max_length

        print(f"Loading tokenizer from: {model_dir}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        print(f"Loading model from: {model_dir}", flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        if self.model.config.num_labels != 2:
            raise ValueError(
                "This interface expects a binary sentiment model with two labels."
            )

        self.model.to(self.device)
        self.model.eval()

        print(f"Inference device: {self.device}", flush=True)
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    def predict(self, review: str):
        """Return a readable result, class probabilities, and token details."""
        review = (review or "").strip()
        if not review:
            raise gr.Error("Please enter a movie review first.")

        tokens_before_truncation = (
            len(self.tokenizer.tokenize(review))
            + self.tokenizer.num_special_tokens_to_add(pair=False)
        )

        encoded = self.tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}

        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        predicted_id = int(torch.argmax(logits, dim=-1).item())
        predicted_label = LABELS[predicted_id]
        confidence = probabilities[predicted_id]
        tokens_used = int(encoded["attention_mask"].sum().item())
        was_truncated = tokens_before_truncation > self.max_length

        result = (
            f"## {predicted_label}\n"
            f"**Confidence: {100 * confidence:.2f}%**"
        )
        class_probabilities = {
            "NEGATIVE": float(probabilities[0]),
            "POSITIVE": float(probabilities[1]),
        }
        details = {
            "tokens_before_truncation": tokens_before_truncation,
            "tokens_used": tokens_used,
            "maximum_tokens": self.max_length,
            "input_was_truncated": was_truncated,
            "inference_device": str(self.device),
        }

        return result, class_probabilities, details


def build_demo(predictor: SentimentPredictor) -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(title="DistilBERT IMDb Sentiment Classifier") as demo:
        gr.Markdown(
            "# DistilBERT Movie Review Sentiment Classifier\n"
            "Enter a movie review to classify it as **positive** or **negative**. "
            "This is the saved seed-42 model, which achieved **91.14% test "
            "accuracy** on IMDb."
        )

        review_input = gr.Textbox(
            label="Movie review",
            placeholder="Type or paste a movie review here...",
            lines=7,
            max_lines=15,
            autofocus=True,
        )

        with gr.Row():
            analyze_button = gr.Button("Analyze sentiment", variant="primary")
            clear_button = gr.ClearButton(value="Clear")

        result_output = gr.Markdown("## Waiting for a review")
        probabilities_output = gr.Label(
            label="Class probabilities",
            num_top_classes=2,
        )
        details_output = gr.JSON(label="Technical details")

        clear_button.add(
            [review_input, result_output, probabilities_output, details_output]
        )

        gr.Examples(
            examples=[
                ["This movie was wonderful, emotional, and beautifully acted."],
                ["The story was boring and the acting was terrible."],
                ["The movie started slowly, but the ending was excellent."],
            ],
            inputs=review_input,
            label="Example reviews",
        )

        gr.Markdown(
            "**Note:** Reviews longer than 256 tokens are truncated. Confidence "
            "is the model's probability estimate, not a guarantee that its "
            "prediction is correct."
        )

        analyze_button.click(
            fn=predictor.predict,
            inputs=review_input,
            outputs=[result_output, probabilities_output, details_output],
            api_name="predict_sentiment",
            concurrency_limit=1,
        )
        review_input.submit(
            fn=predictor.predict,
            inputs=review_input,
            outputs=[result_output, probabilities_output, details_output],
            concurrency_limit=1,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private Gradio UI for the trained DistilBERT model"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = SentimentPredictor(
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )
    demo = build_demo(predictor)

    print(f"Starting private Gradio server on {args.host}:{args.port}", flush=True)
    print("Public Gradio sharing is disabled.", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
