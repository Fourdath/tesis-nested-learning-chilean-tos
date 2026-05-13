from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.incremental_dataset import load_task, add_multihot_labels


BETO_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"


def load_beto_tokenizer() -> PreTrainedTokenizerBase:
    """
    Carga el tokenizador de BETO.

    Importante:
    - BETO usa arquitectura tipo BERT.
    - El tokenizer debe ser del mismo checkpoint que el modelo.
    """
    tokenizer = AutoTokenizer.from_pretrained(BETO_MODEL_NAME)
    return tokenizer


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Tokeniza una única cláusula.
    Devuelve tensores listos para pasarlos al modelo.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    tokenized = {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }

    if "token_type_ids" in encoding:
        tokenized["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

    return tokenized


def tokenize_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Tokeniza varias cláusulas a la vez.
    Esta función es útil para entrenamiento por batches.
    """
    encoding = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    tokenized = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }

    if "token_type_ids" in encoding:
        tokenized["token_type_ids"] = encoding["token_type_ids"]

    return tokenized


def prepare_tokenized_records(
    records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """
    Toma registros del dataset incremental y les agrega los tokens.

    Cada registro queda con:
    - text
    - labels
    - multihot_labels
    - input_ids
    - attention_mask
    - token_type_ids, si el tokenizer lo entrega
    """
    records = add_multihot_labels(records)
    tokenized_records = []

    for record in records:
        tokenized = tokenize_text(
            text=record["text"],
            tokenizer=tokenizer,
            max_length=max_length,
        )

        new_record = record.copy()
        new_record["input_ids"] = tokenized["input_ids"]
        new_record["attention_mask"] = tokenized["attention_mask"]

        if "token_type_ids" in tokenized:
            new_record["token_type_ids"] = tokenized["token_type_ids"]

        new_record["labels_tensor"] = torch.tensor(
            record["multihot_labels"],
            dtype=torch.float,
        )

        tokenized_records.append(new_record)

    return tokenized_records


def inspect_tokenization_example() -> None:
    """
    Prueba rápida para verificar que el tokenizer funciona con el dataset incremental.
    """
    tokenizer = load_beto_tokenizer()

    records = load_task(task_id=1, split="train")
    first_record = records[0]

    text = first_record["text"]

    tokenized = tokenize_text(
        text=text,
        tokenizer=tokenizer,
        max_length=512,
    )

    tokens = tokenizer.convert_ids_to_tokens(
        tokenized["input_ids"].tolist()
    )

    print("Tokenizador BETO cargado correctamente.")
    print()
    print("Texto original:")
    print(text)
    print()
    print("Shapes:")
    print(f"input_ids: {tokenized['input_ids'].shape}")
    print(f"attention_mask: {tokenized['attention_mask'].shape}")

    if "token_type_ids" in tokenized:
        print(f"token_type_ids: {tokenized['token_type_ids'].shape}")

    print()
    print("Primeros 30 tokens:")
    print(tokens[:30])

    print()
    print("Primeros 30 input_ids:")
    print(tokenized["input_ids"][:30].tolist())


if __name__ == "__main__":
    inspect_tokenization_example()