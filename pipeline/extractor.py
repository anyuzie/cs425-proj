"""STEP 1 · Preference extractor (DeBERTa).  STATUS: waiting for your model.

Model: microsoft/deberta-v3-base, fine-tuned on data/extractor.jsonl (manual section 4).
Switch: already on ("extractor": "deberta" in configs/system.json). The server will not start until you finish.

How to integrate your model (you fill ONLY load() and predict(); read() below is ours):
  1. Put your checkpoint in models/extractor/ (gitignored). Share it via Drive, with a hash in the folder name.
  2. Create pipeline/extractor_model.py with your model class (the 3-head DeBERTa: direction, strength, reason).
  3. Fill in load() and predict() below, then delete their `raise NotImplementedError` lines.
  4. pip install -r requirements-models.txt
  5. .venv/bin/python -m unittest discover -s tests
  6. Start the server (.venv/bin/uvicorn app.server:app) and open /api/health: "extractor" must read "deberta".

Do not import torch or transformers at the top of this file; import them inside load() and predict().
"""

import json
import math
import re
from pathlib import Path

from .interfaces import Reading, SlotReading

_SEGMENT = re.compile(r"[^.!?\n]+[.!?]*")
JOIN_MAX_WORDS = 6   # pieces of six words or fewer are joined to the sentence before (manual revision 33)
_NOT_FILLED = "pipeline/extractor.py: fill in load() and predict() (docs/INTEGRATION.md)"


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

    def load(self, checkpoint_dir: Path) -> None:
        """TEAMMATE FILLS. Runs once at server start. Keep what predict() needs on self (tokenizer, model)."""
        # ================= PUT YOUR CODE HERE =================
        # import torch
        # from transformers import AutoTokenizer
        # from .extractor_model import MyExtractorModel      # your file: pipeline/extractor_model.py
        # self.tok = AutoTokenizer.from_pretrained(checkpoint_dir)
        # self.model = MyExtractorModel.from_pretrained(checkpoint_dir).eval()
        # ======================================================
        raise NotImplementedError(_NOT_FILLED)

    def predict(self, context: str | None, sentence: str) -> dict:
        """TEAMMATE FILLS. Run the model on ONE sentence and return exactly:

            {"direction": [[p_neutral, p_positive, p_negative], ...],   # 6 lists of 3, types in R I A S E C order,
                                                                       # class order = data/labels.json direction_index
             "strength":  [s, ...],                                    # 6 floats 0..1 (sigmoid), same type order
             "reason":    [p_enjoyment, p_familiarity, p_neither]}     # 3 probabilities, labels.json reason_index order

        Probabilities must already be softmaxed. Return plain Python floats/lists (tensor.tolist()).
        Input: tokenizer(context, sentence) when context is not None, else tokenizer(sentence);
        max_length=128, truncation=True. Wrap the forward pass in torch.no_grad().
        """
        # ================= PUT YOUR CODE HERE =================
        # import torch
        # args = (context, sentence) if context is not None else (sentence,)
        # enc = self.tok(*args, return_tensors="pt", max_length=128, truncation=True)
        # with torch.no_grad():
        #     out = self.model(**enc)
        # return {"direction": out.direction.softmax(-1)[0].tolist(),     # (6, 3)
        #         "strength": out.strength.sigmoid()[0].tolist(),         # (6,)
        #         "reason": out.reason.softmax(-1)[0].tolist()}           # (3,)
        # ======================================================
        raise NotImplementedError(_NOT_FILLED)

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
