"""Run the whole app WITHOUT the trained models (mock extractor and NLI; everything else is real).

From the repo root:
    .venv/bin/python -m devtools.run_mock serve            # browser: http://localhost:8000
    .venv/bin/python -m devtools.run_mock chat             # chat in the terminal
    .venv/bin/python -m devtools.run_mock eval --limit 20  # full vs no_probing on dev profiles (MOCK numbers)

Mock runs check the flow, not quality: the numbers are not evaluation results.
"""
import argparse
import json
import os
from pathlib import Path

import devtools.mocks  # noqa: F401  registers mock-keyword and mock-nli
from pipeline.interfaces import State
from pipeline.system import System

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/system.mock.json"


def serve(port: int) -> None:
    import uvicorn
    os.environ["CM_CONFIG"] = str(CONFIG)
    from app.server import app   # imported after the mocks are registered, in this same process
    print(f"MOCK MODE: mock extractor and NLI. Open http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


def chat() -> None:
    system, state = System.from_config(ROOT / "data", json.loads(CONFIG.read_text())), State()
    print("MOCK MODE (type 'quit' to stop)\nbot:", system.start(state).reply)
    while not state.done:
        text = input("you: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        result = system.step(state, text)
        profile = {t: round(s.value, 2) for t, s in state.slots.items()}
        print(f"  [{result.action}] profile {profile}\nbot:", result.reply)
        for rec in result.recommendations or []:
            print(f"   {rec['rank']}. {rec['title']}")


def evaluate(limit: int) -> None:
    from eval.dialogue import compare
    from recommender.data_loader import read_profiles
    system = System.from_config(ROOT / "data", json.loads(CONFIG.read_text()))
    profiles = read_profiles(ROOT / "data/dev_profiles.jsonl")[:limit]
    bank = json.loads((ROOT / "data/question_bank.json").read_text())
    results = compare(system, profiles, bank)
    print(f"MOCK comparison on {len(profiles)} dev profiles (flow check only, not a result):")
    for arm, data in results["arms"].items():
        s = data["summary"]
        print(f"  {arm:11} hit@5 {s['hit_at_5']:.3f}  MRR {s['mrr']:.3f}  NDCG@10 {s['ndcg_at_10']:.3f}")
    turns = {arm: sum(len(x.get("turns", [])) for x in d["sessions"]) / max(1, len(d["sessions"]))
             for arm, d in results["arms"].items()}
    print("  average turns:", {k: round(v, 1) for k, v in turns.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve").add_argument("--port", type=int, default=8000)
    sub.add_parser("chat")
    sub.add_parser("eval").add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.port)
    elif args.cmd == "chat":
        chat()
    else:
        evaluate(args.limit)


if __name__ == "__main__":
    main()
