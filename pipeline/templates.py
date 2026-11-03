"""Deterministic question wording, including grounded task fallbacks."""


def _pick(options: list[str], index: int) -> str:
    return options[index % len(options)]


def fill(action, payload: dict, question_bank: dict, type_names: dict[str, str]) -> str:
    """action: Action kind string or Action. payload: index plus slot/quote fields."""
    kind = getattr(action, "kind", action)
    effective = payload.get('retry', 'CLARIFY') if kind == 'ABSTAIN' else kind
    if payload.get('task') and effective in ('CLARIFY', 'PROBE'):
        from .task_questions import task_question
        question = task_question(effective, payload)
        return ('Let me ask about that same activity another way. ' + question
                if kind == 'ABSTAIN' and not payload.get('repair') else question)
    if kind == 'ABSTAIN' and payload.get('repair'):
        from .repair import repair_question
        return repair_question(payload)
    index = payload.get("index", 0)
    if kind == "OPENING":
        return _pick(question_bank["OPENING"], index)
    if kind == "ABSTAIN" and payload.get("retry") in ("CHALLENGE", "PROBE"):
        return f"{_pick(question_bank['ABSTAIN'], index)} {fill(payload['retry'], payload, question_bank, type_names)}"
    if kind in ("CLARIFY", "ABSTAIN"):
        question = _pick(question_bank["CLARIFY"][type_names[payload["slot"]]], payload.get("clarify_index", index))
        if kind == "CLARIFY":
            return ('Thanks. More broadly, ' + question if payload.get('broader_task_check') else question)
        return f"{_pick(question_bank['ABSTAIN'], index)} {question}"
    if kind == "CHALLENGE":
        return _pick(question_bank["CHALLENGE"], index).format(old_src=payload["old_src"], new_src=payload["new_src"])
    if kind == "PROBE":
        return _pick(question_bank["PROBE"], index).format(new_src=payload["new_src"])
    if kind == "RECOMMEND":
        reply = ("Here are five occupations to explore based on the preferences you described:"
                 if payload.get('has_task_evidence') is False else _pick(question_bank["RECOMMEND"], index))
        if payload.get('uncertain_slots'):
            reply += ' Some answers remained uncertain, so treat these as starting points to explore.'
        return reply
    raise ValueError(f"unknown action {kind}")


class QuestionBankFiller:
    name = "question-bank"

    def __init__(self, question_bank: dict, type_names: dict[str, str]) -> None:
        self.qb, self.type_names = question_bank, type_names

    def fill(self, action, payload: dict) -> str:
        return fill(action, payload, self.qb, self.type_names)
