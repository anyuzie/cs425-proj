"""Reply text comes only from data/question_bank.json; nothing is generated."""


def _pick(options: list[str], index: int) -> str:
    return options[index % len(options)]


def fill(action, payload: dict, question_bank: dict, type_names: dict[str, str]) -> str:
    """action: Action kind string or Action. payload: index plus slot/quote fields."""
    kind = getattr(action, "kind", action)
    index = payload.get("index", 0)
    if kind == "OPENING":
        return _pick(question_bank["OPENING"], index)
    if kind in ("CLARIFY", "ABSTAIN"):
        question = _pick(question_bank["CLARIFY"][type_names[payload["slot"]]], payload.get("clarify_index", index))
        if kind == "CLARIFY":
            return question
        return f"{_pick(question_bank['ABSTAIN'], index)} {question}"
    if kind == "CHALLENGE":
        return _pick(question_bank["CHALLENGE"], index).format(old_src=payload["old_src"], new_src=payload["new_src"])
    if kind == "PROBE":
        return _pick(question_bank["PROBE"], index).format(new_src=payload["new_src"])
    if kind == "RECOMMEND":
        return _pick(question_bank["RECOMMEND"], index)
    raise ValueError(f"unknown action {kind}")


class QuestionBankFiller:
    name = "question-bank"

    def __init__(self, question_bank: dict, type_names: dict[str, str]) -> None:
        self.qb, self.type_names = question_bank, type_names

    def fill(self, action, payload: dict) -> str:
        return fill(action, payload, self.qb, self.type_names)
