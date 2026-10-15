"""STEP 4 · NLI contradiction check.  STATUS: not built yet.

Replaces: PassthroughConfirmer in pipeline/stubs.py (it says "yes, contradiction" to everything)
Switch on: configs/system.json -> "confirmer": "nli"
Model:     cross-encoder/nli-deberta-v3-base (no training, used as is)

Only runs when step 3 (the rule) finds two readings of one type with opposite signs.
The system keeps the conflict when score > 0.60 (settings.confirm_threshold).
Tune that number on the dev pairs of data/nli_pairs.jsonl; report once on the test pairs.
"""


class NliConfirmer:
    name = "nli"

    def __init__(self, model_name: str) -> None:
        raise NotImplementedError("pipeline/nli.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        # ================= PUT YOUR CODE HERE: load once =================
        #   from sentence_transformers import CrossEncoder
        #   self.model = CrossEncoder(model_name)
        #   self.contradiction_index = <index of the "contradiction" label for this model>
        # ==================================================================

    def score(self, old_src: str, new_src: str) -> float:
        """old_src  the earlier sentence, e.g. "I like bouncing ideas off people"
        new_src  the new sentence,     e.g. "I hate group work"
        Return the probability (0..1) that they contradict each other, as a plain float."""
        # ================= PUT YOUR CODE HERE =================
        # logits = self.model.predict([(old_src, new_src)])  -> softmax -> take the contradiction column
        # ======================================================
        raise NotImplementedError
