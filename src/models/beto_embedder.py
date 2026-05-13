from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, PreTrainedTokenizerBase

from src.data.incremental_dataset import load_task, add_multihot_labels
from src.data.tokenizer_beto import BETO_MODEL_NAME, load_beto_tokenizer, tokenize_batch


class BetoEmbedder:
    def __init__(
        self,
        model_name: str = BETO_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_name = model_name

        self.tokenizer: PreTrainedTokenizerBase = load_beto_tokenizer()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings_from_texts(
        self,
        texts: list[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Recibe una lista de textos y devuelve embeddings de BETO.

        Salida esperada:
        - Tensor de tamaño [batch_size, hidden_size]
        - En BETO base, hidden_size normalmente es 768.
        """
        tokenized = tokenize_batch(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        tokenized = {
            key: value.to(self.device)
            for key, value in tokenized.items()
        }

        with torch.no_grad():
            outputs = self.model(**tokenized)

        embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def get_embeddings_from_records(
        self,
        records: list[dict[str, Any]],
        max_length: int = 512,
    ) -> torch.Tensor:
        texts = [record["text"] for record in records]

        return self.get_embeddings_from_texts(
            texts=texts,
            max_length=max_length,
        )


def test_beto_embeddings() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_task(task_id=1, split="train")
    records = add_multihot_labels(records)

    sample_records = records[:4]
    sample_texts = [record["text"] for record in sample_records]
    sample_labels = [record["multihot_labels"] for record in sample_records]

    embedder = BetoEmbedder(device=device)

    embeddings = embedder.get_embeddings_from_texts(
        texts=sample_texts,
        max_length=512,
    )

    labels_tensor = torch.tensor(sample_labels, dtype=torch.float)

    print("Embeddings generados correctamente con BETO.")
    print(f"Modelo: {BETO_MODEL_NAME}")
    print(f"Device: {embedder.device}")
    print(f"Cantidad de textos: {len(sample_texts)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print()
    print("Ejemplo de texto:")
    print(sample_texts[0])
    print()
    print("Primeros 10 valores del embedding:")
    print(embeddings[0][:10].detach().cpu())


if __name__ == "__main__":
    test_beto_embeddings()