from __future__ import annotations

import torch

from src.data.incremental_dataset import (
    load_task,
    add_multihot_labels,
    get_num_labels,
)
from src.data.tokenizer_beto import (
    load_beto_tokenizer,
    tokenize_batch,
    BETO_MODEL_NAME,
)
from src.models.transformer_classifier import BetoMultiLabelClassifier


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_beto_tokenizer()

    records = load_task(task_id=1, split="train")
    records = add_multihot_labels(records)

    batch_records = records[:2]

    texts = [record["text"] for record in batch_records]
    labels = [record["multihot_labels"] for record in batch_records]

    tokenized = tokenize_batch(
        texts=texts,
        tokenizer=tokenizer,
        max_length=512,
    )

    tokenized = {
        key: value.to(device)
        for key, value in tokenized.items()
    }

    labels_tensor = torch.tensor(labels, dtype=torch.float).to(device)

    model = BetoMultiLabelClassifier(
        num_labels=get_num_labels(),
        model_name=BETO_MODEL_NAME,
        dropout=0.1,
        freeze_encoder=False,
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            token_type_ids=tokenized.get("token_type_ids"),
            labels=labels_tensor,
        )

    print("Clasificador BETO multi-label probado correctamente.")
    print(f"Device: {device}")
    print(f"input_ids: {tokenized['input_ids'].shape}")
    print(f"attention_mask: {tokenized['attention_mask'].shape}")

    if "token_type_ids" in tokenized:
        print(f"token_type_ids: {tokenized['token_type_ids'].shape}")

    print(f"embeddings: {outputs['embeddings'].shape}")
    print(f"logits: {outputs['logits'].shape}")
    print(f"labels: {labels_tensor.shape}")
    print(f"loss: {outputs['loss'].item():.4f}")


if __name__ == "__main__":
    main()