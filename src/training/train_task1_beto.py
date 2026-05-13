from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import json

import numpy as np
import torch
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "beto_task1_illegal"


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


def build_collate_fn(tokenizer, max_length: int = 512):
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


def get_active_labels(records: list[dict[str, Any]]) -> list[int]:
    active_labels = set()

    for record in records:
        active_labels.update(record["labels"])

    return sorted(active_labels)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def train_one_epoch(
    model: BetoMultiLabelClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Entrenando", leave=False)

    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
        )

        loss = outputs["loss"]
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
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()

    all_predictions = []
    all_targets = []

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluando", leave=False):
        batch = move_batch_to_device(batch, device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
        )

        logits = outputs["logits"]
        loss = outputs["loss"]

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).int()

        all_predictions.append(predictions.cpu())
        all_targets.append(batch["labels"].int().cpu())

        total_loss += loss.item()

    y_pred = torch.cat(all_predictions, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    y_pred_active = y_pred[:, active_labels]
    y_true_active = y_true[:, active_labels]

    metrics = {
        "loss": total_loss / len(dataloader),
        "micro_f1": f1_score(y_true_active, y_pred_active, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true_active, y_pred_active, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true_active, y_pred_active, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true_active, y_pred_active, average="macro", zero_division=0),
        "micro_recall": recall_score(y_true_active, y_pred_active, average="micro", zero_division=0),
        "macro_recall": recall_score(y_true_active, y_pred_active, average="macro", zero_division=0),
    }

    return metrics


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"  loss:            {metrics['loss']:.4f}")
    print(f"  micro_f1:        {metrics['micro_f1']:.4f}")
    print(f"  macro_f1:        {metrics['macro_f1']:.4f}")
    print(f"  micro_precision: {metrics['micro_precision']:.4f}")
    print(f"  macro_precision: {metrics['macro_precision']:.4f}")
    print(f"  micro_recall:    {metrics['micro_recall']:.4f}")
    print(f"  macro_recall:    {metrics['macro_recall']:.4f}")


def main() -> None:
    set_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Modelo base: {BETO_MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Número total de etiquetas: {get_num_labels()}")

    tokenizer = load_beto_tokenizer()

    train_records = add_multihot_labels(load_task(task_id=1, split="train"))
    val_records = add_multihot_labels(load_task(task_id=1, split="val"))
    test_records = add_multihot_labels(load_task(task_id=1, split="test"))

    active_labels = get_active_labels(train_records + val_records + test_records)

    print(f"Task 1 active labels: {active_labels}")
    print(f"Train samples: {len(train_records)}")
    print(f"Val samples:   {len(val_records)}")
    print(f"Test samples:  {len(test_records)}")

    collate_fn = build_collate_fn(tokenizer=tokenizer, max_length=512)

    train_loader = DataLoader(
        TextMultiLabelDataset(train_records),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        TextMultiLabelDataset(val_records),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        TextMultiLabelDataset(test_records),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BetoMultiLabelClassifier(
        num_labels=get_num_labels(),
        model_name=BETO_MODEL_NAME,
        dropout=0.1,
        freeze_encoder=False,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
    )

    num_epochs = 2
    history = []

    best_val_macro_f1 = -1.0

    for epoch in range(1, num_epochs + 1):
        print()
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            active_labels=active_labels,
        )

        print(f"Train loss: {train_loss:.4f}")
        print_metrics("Validación Task 1", val_metrics)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }

        history.append(epoch_record)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]

            checkpoint_path = OUTPUT_DIR / "best_model.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "active_labels": active_labels,
                    "model_name": BETO_MODEL_NAME,
                    "num_labels": get_num_labels(),
                },
                checkpoint_path,
            )

            print(f"Mejor modelo guardado en: {checkpoint_path}")

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        active_labels=active_labels,
    )

    print()
    print_metrics("Test Task 1", test_metrics)

    history_path = OUTPUT_DIR / "training_history.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    metrics_path = OUTPUT_DIR / "test_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2, ensure_ascii=False)

    print()
    print(f"Historial guardado en: {history_path}")
    print(f"Métricas de test guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()