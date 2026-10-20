"""Manual check of the trained extractor (needs torch, transformers and models/extractor/).

Run from the repo root:  .venv/bin/python training/extractor/smoke_test.py
Prints each head's output for six example sentences, plus the readings the pipeline builds from them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.extractor import DebertaExtractor  # noqa: E402

if __name__ == "__main__":
    import json

    checkpoint_dir = ROOT / "models" / "extractor"
    labels_path = ROOT / "data" / "labels.json"

    print("Loading DeBERTa model...")
    extractor = DebertaExtractor(checkpoint_dir, labels_path)
    print(f"Model loaded successfully on {extractor.device}")

    examples = [
        ("What work do you enjoy?", "I enjoy building and repairing machines."),
        ("What work do you enjoy?", "I love investigating scientific problems."),
        ("What work do you enjoy?", "I enjoy helping and teaching people."),
        ("What work do you enjoy?", "I like organising records and keeping information accurate."),
        ("What work do you enjoy?", "I enjoy creating artwork and expressing my imagination."),
        ("What work do you enjoy?", "I like leading teams and persuading people."),
    ]

    for i, (context, sentence) in enumerate(examples, 1):
        print("\n" + "=" * 60)
        print(f"TEST {i}: {sentence}")

        try:
            result = extractor.predict(context, sentence)
            extractor._check(result)

            print("\nDirection probabilities:")
            for slot, probs in zip(extractor.types, result["direction"]):
                idx = max(range(len(probs)), key=probs.__getitem__)
                direction = extractor.direction_names[idx]
                print(f"  {slot}: {direction} | {[round(p, 4) for p in probs]}")

            print("\nStrength:")
            for slot, score in zip(extractor.types, result["strength"]):
                print(f"  {slot}: {score:.4f}")

            reason_idx = max(
                range(len(result["reason"])),
                key=result["reason"].__getitem__
            )
            print("\nReason probabilities:")
            print({
                extractor.reason_names[j]: round(p, 4)
                for j, p in enumerate(result["reason"])
            })
            print(f"Predicted reason: {extractor.reason_names[reason_idx]}")

            print("\nOutput shape: VALID")
            print("Pipeline reads:", [(r.slot, "+" if r.sign > 0 else "-", round(r.strength, 2), round(r.confidence, 2))
                                      for r in extractor.read(sentence, context).slots])

        except Exception as e:
            print(f"\nTEST FAILED: {type(e).__name__}: {e}")
            raise

    print("\nAll test inputs completed successfully.")

