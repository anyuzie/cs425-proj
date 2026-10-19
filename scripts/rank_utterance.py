"""Interactive, independent-turn extraction and configured top-10 ranking.

Run: .venv/Scripts/python.exe scripts/rank_utterance.py
"""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.extractor import DebertaExtractor
from recommender.ranker import RankingPipeline
from recommender.schemas import TYPES


def rank_utterance(text, extractor, ranker):
    """Adapt one Reading to runtime inputs; no history or confidence merging."""
    reading = extractor.read(text)
    interests = dict.fromkeys(TYPES, 0.0)
    observed = set()
    for slot in reading.slots:
        interests[slot.slot] = slot.value
        observed.add(slot.slot)
    # This independent snapshot contains only the current user's utterance.
    evidence = [{"id": "e1", "turn": 1, "text": text, "status": "active"}]
    response = ranker.rank_with_evidence(
        interests, observed, [text], evidence, top_k=10)
    return reading, interests, observed, response


def show_result(reading, interests, observed, response):
    print("\nRIASEC: " + "  ".join(
        f"{t}={interests[t]:+.2f}" if t in observed else f"{t}=unknown"
        for t in TYPES))
    for slot in reading.slots:
        print(f"  {slot.slot}: confidence={slot.confidence:.4f}, reason={slot.reason}")
    if response.status == "no_signal":
        print(f"No ranking: {response.reason}\n")
        return
    print(f"\nTop {len(response.occupations)} occupations:")
    for item in response.occupations:
        scores = []
        if item.rrf_score is not None:
            scores.append(f"RRF={item.rrf_score:.6f}")
        scores.extend(f"{name}={signal.score:.6f} (rank {signal.rank})"
                      for name, signal in item.signals.items())
        print(f"{item.rank:2}. {item.code} | {item.title}")
        print("    " + " | ".join(scores))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-config", type=Path, default=ROOT / "configs/system.json")
    parser.add_argument("--ranking-config", type=Path, help="Override the system's ranking configuration")
    parser.add_argument("--checkpoint", type=Path, help="Override the extractor checkpoint directory")
    parser.add_argument("--occupations", type=Path, default=ROOT / "data/occupations.csv")
    args = parser.parse_args()
    # Existing ranking variants resolve model/cache paths relative to the repo.
    os.chdir(ROOT)
    try:
        config = json.loads(args.system_config.read_text(encoding="utf-8"))
        if config["components"]["extractor"] != "deberta":
            raise ValueError("This script requires the configured deberta extractor")
        ranking_config = args.ranking_config or Path(config["paths"]["ranking_config"])
        checkpoint = args.checkpoint or Path(config["paths"]["extractor_checkpoint"])
        print(f"Loading ranking configuration: {ranking_config}", flush=True)
        ranker = RankingPipeline(args.occupations, config=ranking_config)
        print(f"Loading extractor: {checkpoint}", flush=True)
        extractor = DebertaExtractor(checkpoint, ROOT / "data/labels.json")
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1
    enabled = [f"{k}={v}" for k, v in ranker.component_identifiers.items() if v]
    print("\nEnabled: " + ", ".join(enabled))
    print(f"Eligible occupations: {ranker.eligible_count}")
    print("Each entry is independent. Scores are raw component scores, not probabilities.")
    print("Enter an utterance, or x to exit.")
    while True:
        try:
            text = input("You> ").strip()
            if text == "x":
                break
            if not text:
                continue
            show_result(*rank_utterance(text, extractor, ranker))
        except (EOFError, KeyboardInterrupt):
            break
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Unable to rank this entry: {error}", file=sys.stderr)
    print("Goodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
