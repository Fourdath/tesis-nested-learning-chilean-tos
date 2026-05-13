from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.data.incremental_dataset import (
    load_task,
    add_multihot_labels,
    get_num_labels,
)
from src.data.tokenizer_beto import load_beto_tokenizer, BETO_MODEL_NAME
from src.models.transformer_classifier import BetoMultiLabelClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "beto_incremental_naive"

TASKS = [
    {"task_id": 1, "task_name": "illegal"},
    {"task_id": 2, "task_name": "dark"},
    {"task_id": 3, "task_name": "gray"},
]

MAX_LENGTH = 512
BATCH_SIZE = 4
NUM_EPOCHS_PER_TASK = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
THRESHOLD = 0.5

# active_task = calcula la pérdida solo sobre las etiquetas de la tarea actual.
# all_labels = calcula la pérdida sobre las 24 etiquetas.
LOSS_MODE = "active_task"


class TextMultiLabelDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        return {
            "text": record["text"],
            "labels": torch.tensor(record["multihot_labels"], dtype=torch.float),
            "task_id": record["task_id"],
        }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(tokenizer, max_length: int = MAX_LENGTH):
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        result = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

        if "token_type_ids" in tokenized:
            result["token_type_ids"] = tokenized["token_type_ids"]

        return result

    return collate_fn


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def get_active_labels(records: list[dict[str, Any]]) -> list[int]:
    active_labels = set()

    for record in records:
        active_labels.update(record["labels"])

    return sorted(active_labels)


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_labels: list[int],
    loss_fn: nn.Module,
) -> torch.Tensor:
    if LOSS_MODE == "active_task":
        return loss_fn(logits[:, active_labels], labels[:, active_labels])

    if LOSS_MODE == "all_labels":
        return loss_fn(logits, labels)

    raise ValueError(f"LOSS_MODE no reconocido: {LOSS_MODE}")


def train_one_epoch(
    model: BetoMultiLabelClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    active_labels: list[int],
) -> float:
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Entrenando", leave=False)

    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )

        logits = outputs["logits"]

        loss = compute_loss(
            logits=logits,
            labels=batch["labels"],
            active_labels=active_labels,
            loss_fn=loss_fn,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: BetoMultiLabelClassifier,
    dataloader: DataLoader,
    device: torch.device,
    active_labels: list[int],
    threshold: float = THRESHOLD,
) -> dict[str, float]:
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    all_predictions = []
    all_targets = []
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluando", leave=False):
        batch = move_batch_to_device(batch, device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )

        logits = outputs["logits"]

        loss = compute_loss(
            logits=logits,
            labels=batch["labels"],
            active_labels=active_labels,
            loss_fn=loss_fn,
        )

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).int()

        all_predictions.append(predictions.cpu())
        all_targets.append(batch["labels"].int().cpu())

        total_loss += loss.item()

    y_pred = torch.cat(all_predictions, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    y_pred_active = y_pred[:, active_labels]
    y_true_active = y_true[:, active_labels]

    return {
        "loss": float(total_loss / len(dataloader)),
        "micro_f1": float(f1_score(y_true_active, y_pred_active, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_active, y_pred_active, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true_active, y_pred_active, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true_active, y_pred_active, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true_active, y_pred_active, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_active, y_pred_active, average="macro", zero_division=0)),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  loss:            {metrics['loss']:.4f}")
    print(f"  micro_f1:        {metrics['micro_f1']:.4f}")
    print(f"  macro_f1:        {metrics['macro_f1']:.4f}")
    print(f"  micro_precision: {metrics['micro_precision']:.4f}")
    print(f"  macro_precision: {metrics['macro_precision']:.4f}")
    print(f"  micro_recall:    {metrics['micro_recall']:.4f}")
    print(f"  macro_recall:    {metrics['macro_recall']:.4f}")


def build_loaders_for_task(
    task_id: int,
    tokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[dict[str, Any]]]]:
    train_records = add_multihot_labels(load_task(task_id=task_id, split="train"))
    val_records = add_multihot_labels(load_task(task_id=task_id, split="val"))
    test_records = add_multihot_labels(load_task(task_id=task_id, split="test"))

    collate_fn = build_collate_fn(tokenizer=tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        TextMultiLabelDataset(train_records),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        TextMultiLabelDataset(val_records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        TextMultiLabelDataset(test_records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    records = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    return train_loader, val_loader, test_loader, records


def save_results_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_forgetting_summary(
    results_rows: list[dict[str, Any]],
    final_task_id: int,
) -> list[dict[str, Any]]:
    summary = []

    eval_task_ids = sorted({row["eval_task_id"] for row in results_rows})

    for eval_task_id in eval_task_ids:
        task_rows = [
            row for row in results_rows
            if row["eval_task_id"] == eval_task_id
        ]

        if not task_rows:
            continue

        best_macro_f1 = max(row["macro_f1"] for row in task_rows)
        best_micro_f1 = max(row["micro_f1"] for row in task_rows)

        final_rows = [
            row for row in task_rows
            if row["after_training_task"] == final_task_id
        ]

        if not final_rows:
            continue

        final_row = final_rows[0]

        summary.append(
            {
                "eval_task_id": eval_task_id,
                "eval_task_name": final_row["eval_task_name"],
                "best_macro_f1": best_macro_f1,
                "final_macro_f1": final_row["macro_f1"],
                "macro_forgetting": best_macro_f1 - final_row["macro_f1"],
                "best_micro_f1": best_micro_f1,
                "final_micro_f1": final_row["micro_f1"],
                "micro_forgetting": best_micro_f1 - final_row["micro_f1"],
            }
        )

    return summary


def main() -> None:
    set_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Modelo base: {BETO_MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Número total de etiquetas: {get_num_labels()}")
    print(f"LOSS_MODE: {LOSS_MODE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Épocas por tarea: {NUM_EPOCHS_PER_TASK}")

    tokenizer = load_beto_tokenizer()

    task_loaders = {}
    active_labels_by_task = {}

    for task in TASKS:
        task_id = task["task_id"]
        task_name = task["task_name"]

        train_loader, val_loader, test_loader, records = build_loaders_for_task(
            task_id=task_id,
            tokenizer=tokenizer,
        )

        active_labels = get_active_labels(
            records["train"] + records["val"] + records["test"]
        )

        task_loaders[task_id] = {
            "task_name": task_name,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "records": records,
        }

        active_labels_by_task[task_id] = active_labels

        print()
        print(f"Task {task_id}: {task_name}")
        print(f"  Train samples: {len(records['train'])}")
        print(f"  Val samples:   {len(records['val'])}")
        print(f"  Test samples:  {len(records['test'])}")
        print(f"  Active labels: {active_labels}")

    model = BetoMultiLabelClassifier(
        num_labels=get_num_labels(),
        model_name=BETO_MODEL_NAME,
        dropout=0.1,
        freeze_encoder=False,
    )

    model.to(device)

    training_history = []
    results_rows = []

    for task in TASKS:
        current_task_id = task["task_id"]
        current_task_name = task["task_name"]

        print()
        print("=" * 80)
        print(f"ENTRENANDO TASK {current_task_id}: {current_task_name}")
        print("=" * 80)

        # En baseline secuencial ingenuo reiniciamos el optimizador por tarea.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        train_loader = task_loaders[current_task_id]["train_loader"]
        val_loader = task_loaders[current_task_id]["val_loader"]
        current_active_labels = active_labels_by_task[current_task_id]

        best_val_macro_f1 = -1.0
        best_checkpoint_path = OUTPUT_DIR / f"best_after_task_{current_task_id}_{current_task_name}.pt"

        for epoch in range(1, NUM_EPOCHS_PER_TASK + 1):
            print()
            print(f"Task {current_task_id} | Epoch {epoch}/{NUM_EPOCHS_PER_TASK}")

            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                active_labels=current_active_labels,
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                active_labels=current_active_labels,
            )

            print(f"Train loss: {train_loss:.4f}")
            print_metrics(f"Validación Task {current_task_id}", val_metrics)

            training_history.append(
                {
                    "task_id": current_task_id,
                    "task_name": current_task_name,
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_metrics": val_metrics,
                }
            )

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "task_id": current_task_id,
                        "task_name": current_task_name,
                        "epoch": epoch,
                        "best_val_macro_f1": best_val_macro_f1,
                        "active_labels": current_active_labels,
                        "model_name": BETO_MODEL_NAME,
                        "num_labels": get_num_labels(),
                    },
                    best_checkpoint_path,
                )

                print(f"Mejor checkpoint de la tarea guardado en: {best_checkpoint_path}")

        # Cargamos el mejor checkpoint de la tarea antes de evaluar acumulativamente.
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print()
        print("-" * 80)
        print(f"EVALUACIÓN ACUMULATIVA DESPUÉS DE TASK {current_task_id}")
        print("-" * 80)

        for eval_task in TASKS:
            eval_task_id = eval_task["task_id"]
            eval_task_name = eval_task["task_name"]

            if eval_task_id > current_task_id:
                continue

            test_loader = task_loaders[eval_task_id]["test_loader"]
            eval_active_labels = active_labels_by_task[eval_task_id]

            test_metrics = evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                active_labels=eval_active_labels,
            )

            print_metrics(
                f"Test Task {eval_task_id} ({eval_task_name}) después de entrenar Task {current_task_id}",
                test_metrics,
            )

            results_rows.append(
                {
                    "after_training_task": current_task_id,
                    "after_training_task_name": current_task_name,
                    "eval_task_id": eval_task_id,
                    "eval_task_name": eval_task_name,
                    "loss": test_metrics["loss"],
                    "micro_f1": test_metrics["micro_f1"],
                    "macro_f1": test_metrics["macro_f1"],
                    "micro_precision": test_metrics["micro_precision"],
                    "macro_precision": test_metrics["macro_precision"],
                    "micro_recall": test_metrics["micro_recall"],
                    "macro_recall": test_metrics["macro_recall"],
                }
            )

    history_path = OUTPUT_DIR / "training_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(training_history, file, indent=2, ensure_ascii=False)

    results_path = OUTPUT_DIR / "incremental_results.csv"
    save_results_csv(results_path, results_rows)

    forgetting_summary = compute_forgetting_summary(
        results_rows=results_rows,
        final_task_id=TASKS[-1]["task_id"],
    )

    forgetting_path = OUTPUT_DIR / "forgetting_summary.csv"
    save_results_csv(forgetting_path, forgetting_summary)

    print()
    print("=" * 80)
    print("ENTRENAMIENTO INCREMENTAL FINALIZADO")
    print("=" * 80)
    print(f"Historial guardado en: {history_path}")
    print(f"Resultados incrementales guardados en: {results_path}")
    print(f"Resumen de olvido guardado en: {forgetting_path}")

    print()
    print("Resumen de olvido:")
    for row in forgetting_summary:
        print(
            f"Task {row['eval_task_id']} ({row['eval_task_name']}): "
            f"macro forgetting = {row['macro_forgetting']:.4f} | "
            f"micro forgetting = {row['micro_forgetting']:.4f}"
        )


if __name__ == "__main__":
    main()