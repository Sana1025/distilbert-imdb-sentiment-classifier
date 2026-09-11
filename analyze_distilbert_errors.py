#!/usr/bin/env python3
"""Extract balanced qualitative errors from a trained IMDb DistilBERT model."""

import argparse
import json
import re
from pathlib import Path

from project_config import DATASET_CACHE, PROJECT_DIR

RUNS_DIR = PROJECT_DIR / "runs"
LABEL_NAMES = {0: "negative", 1: "positive"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create balanced error examples from a trained DistilBERT model"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--examples-per-group", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_excerpt(text, maximum_characters=420):
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if len(text) <= maximum_characters:
        return text
    return text[: maximum_characters - 3].rstrip() + "..."


def main():
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.examples_per_group < 1:
        raise ValueError("--examples-per-group must be positive.")

    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this analysis through a GPU Slurm job."
        )

    run_dir = RUNS_DIR / f"seed_{args.seed}"
    model_dir = run_dir / "best_model"
    metrics_path = run_dir / "metrics.json"
    if not model_dir.is_dir() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Completed seed-{args.seed} model/results were not found in {run_dir}"
        )

    output_dir = args.output_dir or PROJECT_DIR / "error_analysis" / f"seed_{args.seed}"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite only if you intentionally want to replace its files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    max_length = int(saved_metrics["config"]["max_length"])

    print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"Model: {model_dir}")
    print(f"Test maximum length: {max_length}")
    print("Loading the fixed IMDb test set for post-training error analysis...")

    raw_test = load_dataset(
        "stanfordnlp/imdb",
        split="test",
        cache_dir=str(DATASET_CACHE),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    map_workers = args.num_workers if args.num_workers > 0 else None
    tokenized_test = raw_test.map(
        tokenize_batch,
        batched=True,
        num_proc=map_workers,
        remove_columns=["text"],
        desc="Tokenizing test",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        tokenized_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model.to("cuda")
    model.eval()

    logits_parts = []
    label_parts = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating test set"):
            if "labels" in batch:
                labels = batch.pop("labels")
            else:
                labels = batch.pop("label")
            batch = {
                name: tensor.to("cuda", non_blocking=True)
                for name, tensor in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch).logits
            logits_parts.append(logits.float().cpu())
            label_parts.append(labels.cpu())

    logits = torch.cat(logits_parts).numpy()
    labels = torch.cat(label_parts).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=-1)
    confidences = probabilities[np.arange(len(predictions)), predictions]

    accuracy = float(accuracy_score(labels, predictions))
    f1 = float(f1_score(labels, predictions, average="binary"))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    saved_accuracy = float(saved_metrics["test_accuracy"])
    metric_reproduced = abs(accuracy - saved_accuracy) <= (0.5 / len(labels))
    if not metric_reproduced:
        print(
            "Warning: reproduced accuracy differs slightly from the saved metric: "
            f"{accuracy:.8f} versus {saved_accuracy:.8f}."
        )

    neg_to_pos = np.flatnonzero((labels == 0) & (predictions == 1))
    pos_to_neg = np.flatnonzero((labels == 1) & (predictions == 0))

    def make_record(index):
        index = int(index)
        return {
            "test_index": index,
            "true_label": LABEL_NAMES[int(labels[index])],
            "predicted_label": LABEL_NAMES[int(predictions[index])],
            "confidence_percent": float(100.0 * confidences[index]),
            "negative_probability_percent": float(100.0 * probabilities[index, 0]),
            "positive_probability_percent": float(100.0 * probabilities[index, 1]),
            "review_excerpt": clean_excerpt(raw_test[index]["text"], 1000),
        }

    all_error_indices = np.sort(np.concatenate([neg_to_pos, pos_to_neg]))
    all_errors = [make_record(index) for index in all_error_indices]

    rng = np.random.default_rng(2026)

    def select_groups(indices):
        count = min(args.examples_per_group, len(indices))
        descending = indices[np.argsort(confidences[indices])[::-1]]
        ascending = indices[np.argsort(confidences[indices])]
        random_indices = rng.choice(indices, size=count, replace=False)
        return {
            "high_confidence": [make_record(index) for index in descending[:count]],
            "borderline": [make_record(index) for index in ascending[:count]],
            "random": [make_record(index) for index in random_indices],
        }

    balanced_examples = {
        "negative_misclassified_as_positive": select_groups(neg_to_pos),
        "positive_misclassified_as_negative": select_groups(pos_to_neg),
    }

    high_confidence_threshold = 0.90
    summary = {
        "seed": args.seed,
        "test_examples": int(len(labels)),
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": matrix.tolist(),
        "negative_misclassified_as_positive": int(len(neg_to_pos)),
        "positive_misclassified_as_negative": int(len(pos_to_neg)),
        "high_confidence_threshold_percent": 90.0,
        "high_confidence_negative_to_positive": int(
            np.sum(confidences[neg_to_pos] >= high_confidence_threshold)
        ),
        "high_confidence_positive_to_negative": int(
            np.sum(confidences[pos_to_neg] >= high_confidence_threshold)
        ),
        "saved_metric_reproduced": metric_reproduced,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "all_misclassified.json").write_text(
        json.dumps(all_errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "balanced_examples.json").write_text(
        json.dumps(balanced_examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )

    lines = [
        f"DISTILBERT ERROR ANALYSIS — SEED {args.seed}",
        "",
        f"Accuracy: {100.0 * accuracy:.2f}%",
        f"F1: {100.0 * f1:.2f}%",
        f"Negative -> positive errors: {len(neg_to_pos)}",
        f"Positive -> negative errors: {len(pos_to_neg)}",
        (
            "Wrong with >=90% confidence: "
            f"{summary['high_confidence_negative_to_positive']} negative->positive, "
            f"{summary['high_confidence_positive_to_negative']} positive->negative"
        ),
        "",
        "Balanced qualitative examples",
    ]

    for direction, groups in balanced_examples.items():
        lines.extend(["", direction.replace("_", " ").upper()])
        for group_name, records in groups.items():
            lines.extend(["", f"  {group_name.replace('_', ' ').title()} errors"])
            for number, record in enumerate(records, start=1):
                lines.append(
                    f"  {number}. Test index {record['test_index']} | "
                    f"confidence {record['confidence_percent']:.2f}%"
                )
                lines.append(f"     {clean_excerpt(record['review_excerpt'])}")

    report = "\n".join(lines) + "\n"
    (output_dir / "balanced_error_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"All analysis outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
