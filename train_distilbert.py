import argparse
import json
import random
import time
from pathlib import Path

from project_config import DATASET_CACHE, MODEL_CACHE, PROJECT_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


RUNS_DIR = PROJECT_DIR / "runs"
MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_NAME = "stanfordnlp/imdb"
SPLIT_SEED = 2026
LABEL_NAMES = ["negative", "positive"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_metrics(evaluation_prediction):
    logits, labels = evaluation_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


def save_learning_curves(log_history, output_path):
    training_points = [
        item for item in log_history if "loss" in item and "eval_loss" not in item
    ]
    validation_points = [item for item in log_history if "eval_loss" in item]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if training_points:
        axes[0].plot(
            [item.get("epoch", 0) for item in training_points],
            [item["loss"] for item in training_points],
            marker="o",
            markersize=3,
            label="Training",
        )
    if validation_points:
        axes[0].plot(
            [item.get("epoch", 0) for item in validation_points],
            [item["eval_loss"] for item in validation_points],
            marker="o",
            label="Validation",
        )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if validation_points:
        epochs = [item.get("epoch", 0) for item in validation_points]
        axes[1].plot(
            epochs,
            [100.0 * item["eval_accuracy"] for item in validation_points],
            marker="o",
            label="Validation accuracy",
        )
        axes[1].plot(
            epochs,
            [100.0 * item["eval_f1"] for item in validation_points],
            marker="o",
            label="Validation F1",
        )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(matrix, output_path):
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("IMDb test confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], LABEL_NAMES)
    axis.set_yticks([0, 1], LABEL_NAMES)

    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=12,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def to_json_value(value):
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script on a GPU-equipped host or "
            "submit it through a GPU scheduler."
        )

    seed_everything(args.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    run_dir = RUNS_DIR / f"seed_{args.seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            f"Completed results already exist at {metrics_path}. "
            "Choose another seed to avoid overwriting them."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    model_dir = run_dir / "best_model"

    device_name = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    experiment_start = time.time()

    print(f"Device: cuda")
    print(f"GPU: {device_name}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {args.seed}; split seed: {SPLIT_SEED}")

    print("\nLoading IMDb dataset...")
    raw_dataset = load_dataset(DATASET_NAME, cache_dir=str(DATASET_CACHE))
    training_split = raw_dataset["train"].train_test_split(
        test_size=0.20,
        seed=SPLIT_SEED,
        stratify_by_column="label",
    )
    raw_train = training_split["train"]
    raw_validation = training_split["test"]
    raw_test = raw_dataset["test"]

    print(f"Training examples: {len(raw_train):,}")
    print(f"Validation examples: {len(raw_validation):,}")
    print(f"Test examples: {len(raw_test):,} (untouched until final evaluation)")
    print(f"Unsupervised examples: {len(raw_dataset['unsupervised']):,} (not used)")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE))
    map_workers = args.num_workers if args.num_workers > 0 else None

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    print("Tokenizing training and validation data...")
    tokenized_train = raw_train.map(
        tokenize_batch,
        batched=True,
        num_proc=map_workers,
        remove_columns=["text"],
        desc="Tokenizing train",
    )
    tokenized_validation = raw_validation.map(
        tokenize_batch,
        batched=True,
        num_proc=map_workers,
        remove_columns=["text"],
        desc="Tokenizing validation",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {label: index for index, label in id2label.items()}

    print("Loading pretrained DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        cache_dir=str(MODEL_CACHE),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Model parameters: {parameter_count:,}")

    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        # Transformers 5.x interprets a fractional warmup_steps value as a ratio.
        warmup_steps=0.10,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("\nStarting fine-tuning...")
    training_result = trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))

    print("\nBest validation checkpoint selected.")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best validation accuracy: {100.0 * trainer.state.best_metric:.2f}%")

    print("\nTokenizing and evaluating the untouched test set exactly once...")
    tokenized_test = raw_test.map(
        tokenize_batch,
        batched=True,
        num_proc=map_workers,
        remove_columns=["text"],
        desc="Tokenizing test",
    )
    test_output = trainer.predict(tokenized_test, metric_key_prefix="test")
    predictions = np.argmax(test_output.predictions, axis=-1)
    labels = test_output.label_ids
    probabilities = torch.softmax(
        torch.from_numpy(test_output.predictions), dim=-1
    ).numpy()
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    report = classification_report(
        labels,
        predictions,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    correct = int((predictions == labels).sum())
    total = int(len(labels))
    test_accuracy = float(test_output.metrics["test_accuracy"])
    test_f1 = float(test_output.metrics["test_f1"])
    total_seconds = time.time() - experiment_start
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    mistakes = []
    wrong_indices = np.flatnonzero(predictions != labels)[:50]
    for index in wrong_indices:
        predicted_label = int(predictions[index])
        mistakes.append(
            {
                "test_index": int(index),
                "true_label": LABEL_NAMES[int(labels[index])],
                "predicted_label": LABEL_NAMES[predicted_label],
                "confidence_percent": float(probabilities[index, predicted_label] * 100),
                "review_excerpt": raw_test[int(index)]["text"][:500],
            }
        )

    save_learning_curves(trainer.state.log_history, run_dir / "learning_curves.png")
    save_confusion_matrix(matrix, run_dir / "confusion_matrix.png")

    with open(run_dir / "misclassified_examples.json", "w", encoding="utf-8") as file:
        json.dump(mistakes, file, indent=2, ensure_ascii=False)
    with open(run_dir / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(to_json_value(report), file, indent=2)
    with open(run_dir / "log_history.json", "w", encoding="utf-8") as file:
        json.dump(to_json_value(trainer.state.log_history), file, indent=2)

    metrics = {
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "training_examples": len(raw_train),
        "validation_examples": len(raw_validation),
        "test_examples": len(raw_test),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_accuracy": float(trainer.state.best_metric),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_correct": correct,
        "test_total": total,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "model_parameters": parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "training_metrics": training_result.metrics,
        "test_metrics": test_output.metrics,
        "total_seconds": total_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "config": vars(args),
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(to_json_value(metrics), file, indent=2)

    print("\nSample test predictions:")
    for index in range(10):
        predicted_label = int(predictions[index])
        print(
            f"{index + 1:2d}. True: {LABEL_NAMES[int(labels[index])]:8s} | "
            f"Predicted: {LABEL_NAMES[predicted_label]:8s} | "
            f"Confidence: {probabilities[index, predicted_label] * 100:.2f}%"
        )

    print("\nFinal result")
    print(f"Test accuracy: {100.0 * test_accuracy:.2f}% ({correct}/{total})")
    print(f"Test F1: {100.0 * test_f1:.2f}%")
    print(f"Peak allocated GPU memory: {peak_gpu_memory_mb:.1f} MiB")
    print(f"Total experiment time: {total_seconds:.1f} seconds")
    print(f"All outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()
