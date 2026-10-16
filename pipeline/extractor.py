"""STEP 1 · Preference extractor (DeBERTa).  STATUS: not built yet.

Replaces: LexiconExtractor in pipeline/stubs.py
Switch on: configs/system.json -> "extractor": "deberta"
Model:     microsoft/deberta-v3-base, fine-tuned on data/extractor.jsonl (manual section 4)
"""

from pathlib import Path

from .interfaces import Reading, SlotReading  # noqa: F401  (you will build these)


class DebertaExtractor:
    name = "deberta"

    def __init__(self, checkpoint_dir: Path) -> None:
        raise NotImplementedError("pipeline/extractor.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        # ================= PUT YOUR CODE HERE: load once =================
        # Runs once when the server starts. Load the tokenizer and the trained checkpoint, e.g.
        #   from transformers import AutoTokenizer
        #   self.tok = AutoTokenizer.from_pretrained(checkpoint_dir)
        #   self.model = <your 3-head model>.from_pretrained(checkpoint_dir).eval()
        # ==================================================================

    def read(self, text: str, context: str | None = None, expected_slot: str | None = None) -> Reading:
        """text          the user's message
        context       the bot question being answered (None on the first message)
        expected_slot the letter that question was about, e.g. "C" (None if not a question)

        Return Reading(text, slots) with one SlotReading per type that has a stance:
            SlotReading(slot="I", sign=+1, strength=0.55, confidence=0.91,
                        reason="enjoyment", source="<the sentence it came from, verbatim>")
        slot = letter R I A S E C · sign = +1 / -1 · strength and confidence in 0..1
        reason = "enjoyment" | "familiarity" | "neither". Types read as neutral are left out.
        """
        # ================= PUT YOUR CODE HERE: one message =================
        # 1. inputs = self.tok(context, text) if context else self.tok(text)   (max 128 tokens)
        # 2. run the model -> direction (6x3), strength (6), reason (3)
        # 3. for each type not predicted "neutral": sign from direction, confidence = softmax prob
        #    of the chosen class, strength from the strength head
        # 4. source: the sentence of `text` the stance came from (split on . ! ?), kept word for word
        # ===================================================================
        raise NotImplementedError
