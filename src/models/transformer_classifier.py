from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from src.data.tokenizer_beto import BETO_MODEL_NAME


class BetoMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        model_name: str = BETO_MODEL_NAME,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**inputs)

        # Usamos [CLS] desde last_hidden_state, no pooler_output.
        embeddings = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(self.dropout(embeddings))

        result = {
            "logits": logits,
            "embeddings": embeddings,
        }

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result