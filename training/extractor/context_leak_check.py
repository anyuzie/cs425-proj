"""Does the bot's question leak into the extractor's reading of the user's answer?

Runs each failing scenario reply twice through the real model: on its own, and with the bot's
question attached (as the app does). If a stance appears only WITH the question, the question is
leaking into the reading.

Run from the repo root (needs torch, transformers and the checkpoint in models/extractor/):
    .venv/bin/python training/extractor/context_leak_check.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.extractor import Extractor  # noqa: E402

bank = json.loads((ROOT / "data/question_bank.json").read_text(encoding="utf-8"))
challenge = bank["CHALLENGE"][0].format(old_src="I love repairing machines and working with tools.",
                                        new_src="I hate repairing machines and working with tools.")
clarify_c = bank["CLARIFY"]["Conventional"][0]

CASES = [
    ("#11 unrelated answer to the tools challenge", challenge, "I enjoy analysing scientific research data."),
    ("#10 uncertain answer to the tools challenge", challenge, "hmm"),
    ("#12 neutral answer to a Conventional question", clarify_c, "I cannot decide."),
    ("#12 neutral answer to a Conventional question", clarify_c, "I am not sure."),
    ("control: short choice reply (needs the question)", clarify_c, "the first one"),
]


def show(reading):
    return ", ".join(f"{r.slot}{'+' if r.sign > 0 else '-'} (strength {r.strength:.2f}, conf {r.confidence:.2f})"
                     for r in reading.slots) or "no stance"


extractor = Extractor(ROOT / "models/extractor", ROOT / "data/labels.json")
for name, question, reply in CASES:
    alone, paired = extractor.read(reply), extractor.read(reply, context=question)
    leaked = sorted({r.slot for r in paired.slots} - {r.slot for r in alone.slots})
    print(f"\n{name}\n  reply:              {reply!r}\n  question:           {question[:90]}...")
    print(f"  reply alone:        {show(alone)}\n  reply + question:   {show(paired)}")
    print(f"  only with question: {', '.join(leaked) if leaked else '—'}")
print("\nIf the first four cases show stances 'only with question', the question is leaking into the reading.")
print("The last case SHOULD gain a stance with the question; that is what the context is for.")
