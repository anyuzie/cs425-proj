"""DeBERTa checkpoint adapter for the M5 extractor -> tracker -> ranking boundary.

The trained sequence-classification checkpoint emits 45 logits: direction (6x3),
reason (3), and strength (6x4). Calibration comes from config.extractor. Runtime
SlotReading values use the existing label indices and discrete strength mapping.
Configure paths.extractor_checkpoint in configs/system.json; local files only.
Heavy dependencies and the checkpoint are loaded once, on construction.
"""

# Original teammate hand-off instructions above are preserved verbatim from main.
# Current adapter: load()/predict() are implemented using Hugging Face AutoModel,
# so a separate extractor_model.py is not required by this implementation.
# Supply the complete local checkpoint (config, tokenizer and trained weights),
# directly in models/extractor/ or at configs/system.json paths.extractor_checkpoint.
# The adapter expects 45 semantically labelled logits and config.extractor calibration.
# See reports/ranking/m5/EXTRACTOR_INTEGRATION.md for implementation and validation status.

import json
import math
import re
from pathlib import Path

from .interfaces import Reading, SlotReading

from transformers import (
AutoTokenizer,
AutoModelForSequenceClassification,
)
from pathlib import Path
import torch
from safetensors.torch import load_model
from transformers import AutoConfig
import torch.nn as nn
from transformers import AutoModel

_SEGMENT = re.compile(r"[^.!?\n]+[.!?]*")
JOIN_MAX_WORDS = 6   # pieces of six words or fewer are joined to the sentence before (manual revision 33)


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, float(x)))
class CustomConcatWithLayer(nn.Module):
    def __init__(self, custom_layers_n, model_name, dropout=0.1):
        super().__init__()
        try:  # some models have a pooling layer (roberta), others don't (deberta)
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        except TypeError:
            self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.config = self.encoder.config
        
        self.dropout = nn.Dropout(dropout)
        # input to the classifier is larger because of concatenation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * custom_layers_n, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 45),
        )

    def forward(self, input_ids, attention_mask, custom_layers_n):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # concatenate the last n_layers for pooling
        hs = outputs.hidden_states[-custom_layers_n:]
        mask = attention_mask.unsqueeze(-1).to(hs[0].dtype)       # (B, T, 1)
        denom = mask.sum(1).clamp(min=1e-9)                       # (B, 1)

        pooled = torch.cat([(h * mask).sum(1) / denom for h in hs], dim=-1)  # (B, n * H)

        return self.classifier(self.dropout(pooled))
    
class Extractor:
    name = "deberta"

    def __init__(self, checkpoint_dir: Path, labels_path: Path) -> None:
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.types = labels["types"]                                                     # R I A S E C
        self.direction_names = {i: n for n, i in labels["direction_index"].items()}      # neutral positive negative
        self.direction_index_neutral = labels["direction_index"]["neutral"]
        self.reason_names = {i: n for n, i in labels["reason_index"].items()}            # enjoyment familiarity neither
        self.neutral_idx = labels["direction_index"]["neutral"]
        self.config = AutoConfig.from_pretrained(checkpoint_dir)       # reads saved config.json
        self.cfg = self.config.extractor
        
        self.load(Path(checkpoint_dir))
        
    ###################################################################
    # Load Tokenizer, Model
    ###################################################################
    def load(self, checkpoint_dir: Path) -> None:
        if not checkpoint_dir.is_dir() or not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Extractor checkpoint missing at {checkpoint_dir.resolve()}. "
                "Place the trained config.json, tokenizer files and weights directly in this folder, "
                "or set paths.extractor_checkpoint in the system config (docs/INTEGRATION.md).")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

        self.tok = AutoTokenizer.from_pretrained(checkpoint_dir)

        if self.cfg["custom_name"]!= None:                                           # custom head model
            model_cls = globals()[self.cfg["custom_name"]]
            model = model_cls(self.cfg["custom_layers_pool_conc"], self.cfg["model_name"])
            load_model(model, str(checkpoint_dir / "model.safetensors"))
        else:                                                        # plain HF classifier
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

        self.model = model.to(self.device).eval() 

        if not isinstance(self.cfg, dict):
            raise ValueError("Extractor checkpoint config requires extractor calibration settings")
        if type(self.cfg.get("max_len")) is not int or self.cfg["max_len"] < 4:
            raise ValueError("Extractor max_len must be an integer >= 4")
        if getattr(self.config, "num_labels", None) != 45:
            raise ValueError("Extractor checkpoint must output 45 logits: direction 18, reason 3, strength 24")
        for key in ("T_slot", "T_reason", "T_strength"):
            value = self.cfg.get(key, 1.0 if key == "T_strength" else None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Extractor {key} must be a finite positive temperature")
        self._configure_heads(self.config)

    def _configure_heads(self, config):
        """Translate checkpoint logit labels into the shared labels.json ordering."""
        labels = getattr(config, "id2label", {})
        expected = ({f"{t}:{name}" for t in self.types for name in self.direction_names.values()}
                    | set(self.reason_names.values())
                    | {f"{t}:strength:{name}" for t in self.types
                       for name in ("inactive", "dontmind", "mild", "strong")})
        if len(labels) != 45 or set(labels.values()) != expected:
            raise ValueError("Extractor id2label must explicitly name all 45 direction/reason/strength logits")
        indices = {name: int(i) for i, name in labels.items()}
        if set(indices.values()) != set(range(45)):
            raise ValueError("Extractor id2label indices must be exactly 0..44")
        self.direction_indices = [indices[f"{t}:{self.direction_names[i]}"] for t in self.types for i in range(3)]
        self.reason_indices = [indices[self.reason_names[i]] for i in range(3)]
        self.strength_indices = [indices[f"{t}:strength:{name}"] for t in self.types
                                 for name in ("dontmind", "mild", "strong")]

    def predict(self, context: str | None, sentence: str) -> dict:
        """Run the trained model on ONE sentence and return exactly:

            {"direction": [[p_neutral, p_positive, p_negative], ...],   # 6 lists of 3, types in R I A S E C order,
                                                                       # class order = data/labels.json direction_index
             "strength":  [s, ...],                                    # 6 mapped strengths 0..1, same type order
             "reason":    [p_enjoyment, p_familiarity, p_neither]}     # 3 probabilities, labels.json reason_index order

        Probabilities must already be softmaxed. Return plain Python floats/lists (tensor.tolist()).
        Input: tokenizer(context, sentence) when context is not None, else tokenizer(sentence);
        max_length=128, truncation=True. Wrap the forward pass in torch.no_grad().
        """
        import torch

        final_dict = {}
        args = (context, sentence) if context else (sentence,)
        enc = self.tok(*args, truncation=True, max_length=self.cfg["max_len"],
                    return_tensors="pt", return_token_type_ids=False).to(self.device)
        if self.cfg["custom_name"] != None:
            lg = self.model(**enc, custom_layers_n = self.cfg["custom_layers_pool_conc"])[0].float().cpu()
        else:
            lg = self.model(**enc, custom_layers_n = self.cfg["custom_layers_pool_conc"]).logits[0].float().cpu()                                 # 45 scores
    
        sp = torch.softmax(lg[self.direction_indices].view(6, 3) / self.cfg["T_slot"], -1).tolist()
        rp = torch.softmax(lg[self.reason_indices] / self.cfg["T_reason"], -1).tolist()
        # Retain the trained active-class mapping; inactive strength is omitted.
        stp = torch.softmax(lg[self.strength_indices].view(6, 3) / self.cfg.get("T_strength", 1.0), -1).tolist()

        strength_scores = [0, 0, 0, 0, 0, 0] #initialised to 0 first
        strength_mapping = [0.2, 0.55, 1] # maps to: dontmind / mild / strong
        for i in range(len(sp)):
            pol = sp[i]
            if max(range(len(pol)), key=pol.__getitem__) == self.direction_index_neutral:
                continue #if polarity highest score is neutral, strength score is 0
            else:
                strength_probs = stp[i]
                strength_scores[i] =  strength_mapping[strength_probs.index(max(strength_probs))] #get the highest probability strength score
                
        final_dict["direction"] = sp
        final_dict["strength"] = strength_scores
        final_dict["reason"] = rp

        return final_dict

    # ---- below: ours, do not edit ----

    @staticmethod
    def sentences(text: str) -> list[str]:
        """Split on . ! ? and newlines; pieces of six words or fewer join the sentence before. Verbatim spans."""
        spans: list[list[int]] = []
        for m in _SEGMENT.finditer(text):
            if not m.group().strip(" \t\r.!?"):
                continue
            if spans and len(m.group().split()) <= JOIN_MAX_WORDS:
                spans[-1][1] = m.end()
            else:
                spans.append([m.start(), m.end()])
        return [text[a:b].strip() for a, b in spans]

    def _check(self, out: dict) -> None:
        """Fail loudly, naming the key, if predict() returned the wrong shape."""
        nt, nd, nr = len(self.types), len(self.direction_names), len(self.reason_names)
        # key -> (rows expected, length of each row, description for the error)
        want = {"direction": (nt, nd, f"{nt} lists of {nd} probabilities"),
                "strength": (1, nt, f"{nt} floats"),
                "reason": (1, nr, f"{nr} probabilities")}
        if not isinstance(out, dict):
            raise ValueError(f"predict() must return a dict with keys {sorted(want)}, got {type(out).__name__}")
        for key, (n_rows, row_len, desc) in want.items():
            if key not in out:
                raise ValueError(f"predict() result is missing key {key!r}; expected {desc}")
            rows = out[key] if key == "direction" else [out[key]]
            try:
                ok = (len(rows) == n_rows
                      and all(len(row) == row_len for row in rows)
                      and all(math.isfinite(float(x)) for row in rows for x in row))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ValueError(f"predict() key {key!r} has the wrong shape or non-numeric values; expected {desc}")

    def read(self, text: str, context: str | None = None, expected_slot: str | None = None) -> Reading:
        # expected_slot is accepted for interface compatibility and unused: the model already sees the question as context.
        best: dict[str, SlotReading] = {}
        for sentence in self.sentences(text):
            out = self.predict(context, sentence)
            self._check(out)
            reason_p = [float(p) for p in out["reason"]]
            ri = max(range(len(reason_p)), key=reason_p.__getitem__)
            for t, probs, strength in zip(self.types, out["direction"], out["strength"]):
                probs = [float(p) for p in probs]
                ci = max(range(len(probs)), key=probs.__getitem__)
                name = self.direction_names[ci]
                if name == "neutral":
                    continue
                r = SlotReading(t, 1 if name == "positive" else -1, _clamp(strength), _clamp(probs[ci]),
                                self.reason_names[ri], sentence, _clamp(reason_p[ri]))
                if t not in best or r.confidence > best[t].confidence:
                    best[t] = r
        return Reading(text, list(best.values()))