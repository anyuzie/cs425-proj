"""Optional T5 wording with exact Python-owned quotations and bank fallback."""

import json
import logging
from pathlib import Path

from .templates import QuestionBankFiller, _pick


class T5Filler(QuestionBankFiller):
    name = "t5"

    def __init__(self, question_bank, type_names, checkpoint):
        super().__init__(question_bank, type_names)
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        checkpoint = Path(checkpoint)
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"T5 checkpoint missing at {checkpoint.resolve()}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True).eval()

    def generate(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if tokens.input_ids.shape[1] > 256:
            return ""
        with self.torch.inference_mode():
            output = self.model.generate(**tokens, max_new_tokens=64, do_sample=False,
                                         num_beams=2, no_repeat_ngram_size=3)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def fill(self, action, payload):
        fallback = super().fill(action, payload)
        kind = getattr(action, "kind", action)
        if kind == 'ABSTAIN' and payload.get('repair'):
            return fallback
        if kind in ("OPENING", "RECOMMEND"):
            return fallback
        retry = kind == "ABSTAIN"
        kind = payload.get("retry", "CLARIFY") if retry else kind
        if kind == "CLARIFY":
            reference = super().fill("CLARIFY", payload)
            context = []
        elif kind == "PROBE":
            reference = "Is this something you would enjoy, or mainly something you have experience with?"
            context = [payload["new_src"]]
        elif kind == "CHALLENGE":
            reference = "Which statement better describes the work you would actually want to do?"
            context = [payload["old_src"], payload["new_src"]]
        else:
            return fallback
        previous = payload.get("previous_question", "") if retry else ""
        prompt = "career question: " + json.dumps(dict(
            action=kind, type=self.type_names[payload["slot"]], reference=reference,
            context=context, retry=retry, previous_question=previous,
            style=("plain", "direct", "reflective")[payload.get("index", 0) % 3]), ensure_ascii=False)
        try:
            question = self.generate(prompt)
        except (RuntimeError, ValueError, OSError):
            logging.getLogger(__name__).exception("T5 failed; using question-bank template")
            return fallback
        if not self.valid(question, kind, previous):
            return fallback
        prefix = _pick(self.qb["ABSTAIN"], payload.get("index", 0)) + " " if retry else ""
        if kind == "PROBE":
            prefix += f'You said "{context[0]}". '
        elif kind == "CHALLENGE":
            prefix += f'Earlier you said "{context[0]}" and then "{context[1]}". '
        return prefix + question

    @staticmethod
    def valid(text, kind, previous=""):
        if not isinstance(text, str) or not 5 <= len(text.split()) <= 60:
            return False
        if not text.endswith("?") or text.count("?") != 1 or any(c in text for c in '\n\r"{}<>'):
            return False
        if previous and " ".join(text.lower().split()) in " ".join(previous.lower().split()):
            return False
        return kind != "CHALLENGE" or any(w in text.lower() for w in ("which", "closer", "prefer"))
