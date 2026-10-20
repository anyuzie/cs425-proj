"""Manual section 5, step 4: pretrained NLI on candidate source sentences.

Label evidence: https://huggingface.co/cross-encoder/nli-deberta-v3-base/blob/main/config.json
Published order: contradiction, entailment, neutral. Inspect the loaded config;
never assume anonymous LABEL_n mappings have that order.
"""

import math

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


class NliConfirmer:
    name = "nli"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if set(labels) != {0, 1, 2} or set(labels.values()) != {
            "contradiction", "entailment", "neutral"
        }:
            raise ValueError(f"Unverified NLI output label mapping: {labels!r}")
        self.label_order = [labels[i] for i in range(3)]
        self.contradiction_index = self.label_order.index("contradiction")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def score(self, old_src: str, new_src: str) -> float:
        """Return P(contradiction), separately from extractor confidence."""
        return self.probabilities(old_src, new_src)["contradiction"]

    def probabilities(self, old_src: str, new_src: str) -> dict[str, float]:
        """Return all three named NLI probabilities from a single forward pass."""
        import torch

        features = self.tokenizer(old_src, new_src, padding=True, truncation=True,
                                  max_length=512, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**features).logits
            if tuple(logits.shape) != (1, 3):
                raise ValueError(f"Expected one NLI row with three logits, got {logits.shape}")
            values = torch.softmax(logits.float(), dim=-1)[0].tolist()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("NLI returned a non-finite probability")
        return {label: float(values[i]) for i, label in enumerate(self.label_order)}
