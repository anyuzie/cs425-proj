"""STEP 1 · Preference extractor (DeBERTa).  STATUS: model integrated.

Model: microsoft/deberta-v3-base fine-tuned on data/extractor.jsonl (manual section 4), loaded from
models/extractor/ (gitignored; get the checkpoint from the team Drive and unzip it so config.json sits
directly in that folder). Config: "extractor": "deberta" in configs/system.json.

load() and predict() are the model owner's code. read() and the rest are the pipeline's: it splits the
message into sentences, calls predict() once per sentence and turns the probabilities into SlotReadings.
The class order is taken from the model's own id2label (e.g. "R:negative", "R:neutral", "R:positive"),
so predict() can return probabilities in the model's order; data/labels.json is only the fallback.

torch and transformers are imported inside load() and predict(), so the tests run without them.
Manual check with the real model: .venv/bin/python training/extractor/smoke_test.py
"""

import json
import math
import re
from pathlib import Path

from .interfaces import Reading, SlotReading


_SEGMENT = re.compile(r"[^.!?\n]+[.!?]*")
JOIN_MAX_WORDS = 6   # pieces of six words or fewer are joined to the sentence before (manual revision 33)
_NO_CHECKPOINT = ("no trained extractor at {path}: get the checkpoint from the team Drive and unzip it so "
                  "config.json sits directly in models/extractor/ (docs/INTEGRATION.md)")


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


class DebertaExtractor:
    name = "deberta"

    def __init__(self, checkpoint_dir: Path, labels_path: Path) -> None:
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.types = labels["types"]                                                     # R I A S E C
        self.direction_names = {i: n for n, i in labels["direction_index"].items()}      # neutral positive negative
        self.reason_names = {i: n for n, i in labels["reason_index"].items()}            # enjoyment familiarity neither
        self.load(Path(checkpoint_dir))
        self._adopt_model_labels()

    def _adopt_model_labels(self) -> None:
        """Use the class order the model was trained with, from its config.id2label, if it has one.

        Direction labels look like "R:negative"; reason labels are bare names like "enjoyment".
        predict() returns rows in that order, so read() must index them the same way.
        """
        config = getattr(getattr(self, "model", None), "config", None)
        id2label = getattr(config, "id2label", None) or {}
        labels = [id2label.get(i, id2label.get(str(i), "")) for i in range(len(id2label))]
        direction = [lab.split(":", 1)[1] for lab in labels if lab.startswith(f"{self.types[0]}:")
                     and lab.split(":", 1)[1] in ("negative", "neutral", "positive")]
        reasons = [lab for lab in labels if lab in ("enjoyment", "familiarity", "neither")]
        if sorted(direction) == ["negative", "neutral", "positive"]:
            for t in self.types:   # every type must use the same order, since read() applies one order to all rows
                if [lab.split(":", 1)[1] for lab in labels if lab.startswith(f"{t}:")
                        and lab.split(":", 1)[1] in ("negative", "neutral", "positive")] != direction:
                    raise ValueError(f"model id2label gives type {t} a different class order than {self.types[0]}")
            self.direction_names = dict(enumerate(direction))
        if sorted(reasons) == ["enjoyment", "familiarity", "neither"]:
            self.reason_names = dict(enumerate(reasons))

    ###################################################################
    # Load Tokenizer, Model
    ###################################################################
    def load(self, checkpoint_dir: Path) -> None:
        if not (Path(checkpoint_dir) / "config.json").is_file():
            raise FileNotFoundError(_NO_CHECKPOINT.format(path=checkpoint_dir))
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(self.device).eval()
        self.cfg = self.model.config.extractor

    def predict(self, context: str | None, sentence: str) -> dict:
        """Run the model on ONE sentence. Returns plain Python lists:

            {"direction": 6 lists of 3 probabilities, types in R I A S E C order, classes in the model's
                          id2label order (this model: negative, neutral, positive),
             "strength":  6 values: 0 for a neutral type, else 0.2 / 0.55 / 1.0 (dontmind / mild / strong),
             "reason":    3 probabilities (enjoyment, familiarity, neither)}
        """
        import torch

        final_dict = {}
        args = (context, sentence) if context else (sentence,)
        enc = self.tok(*args, truncation="only_first" if context else True, max_length=self.cfg["max_len"],
                    return_tensors="pt", return_token_type_ids=False).to(self.device)
        with torch.no_grad():
            lg = self.model(**enc).logits[0].float().cpu()                              # 45 scores
    
        sp  = torch.softmax(lg[:18].view(6, 3) / self.cfg["T_slot"], -1).tolist()                # [6, 3]  polarity per type
        rp  = torch.softmax(lg[18:21] / self.cfg["T_reason"], -1).tolist()                       # [3]     reason
        # strength: 6 types x 4 classes; drop the "inactive" column, keep dontmind / mild / strong
        stp = torch.softmax(lg[21:45].view(6, 4)[:, 1:] / self.cfg.get("T_strength", 1.0), -1).tolist()

        strength_scores = [0, 0, 0, 0, 0, 0] #initialised to 0 first
        strength_mapping = [0.2, 0.55, 1] #maps to: dontmind / mild / strong
        # position of "neutral" in the model's class order (1 for this model: negative, neutral, positive)
        neutral = next(i for i, name in self.direction_names.items() if name == "neutral")
        for i in range(len(sp)):
            pol = sp[i]
            if max(pol) == pol[neutral]:   # check if the maximum probability is neutral
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
