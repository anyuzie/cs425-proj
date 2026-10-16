"""Merge team-labelled real sentences into extractor.jsonl.
Usage: python import_real.py labeller1.csv [labeller2.csv]
With two files (the same sentences, labelled independently) it prints the agreement and Cohen's kappa per type.
With one agreed file it appends rows: 50 random sentences go to split 'real_test' (never trained on), the rest to 'train'."""
import csv, json, sys, random, os
from sklearn.metrics import cohen_kappa_score
T = "RIASEC"; D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SV = {"strong": 1.0, "mild": 0.55, "dontmind": 0.2}
def lab(row, t): return {"+": "positive", "-": "negative"}.get(row[t].strip(), "neutral")
rows = [list(csv.DictReader(open(f))) for f in sys.argv[1:]]
if len(rows) == 2:
    a, b = rows
    for t in T:
        print(t, "kappa", round(cohen_kappa_score([lab(x, t) for x in a], [lab(x, t) for x in b]), 3))
    print("reason kappa", round(cohen_kappa_score([x["reason"] for x in a], [x["reason"] for x in b]), 3))
    sys.exit()
random.seed(425); data = rows[0]; random.shuffle(data)
out = open(os.path.join(D, "extractor.jsonl"), "a")
for i, x in enumerate(data):
    slots = {t: lab(x, t) for t in T}
    strength = {t: (x[t + "_str"] or "mild") for t in T if slots[t] != "neutral"}
    strength_label = strength
    strength = {t: SV[v] for t, v in strength_label.items()}
    out.write(json.dumps({"text": x["text"], "context": x["context"], "slots": slots, "strength": strength, "strength_label": strength_label,
                          "reason": x["reason"], "seed_id": f"REAL_{i:03d}", "kind": "real",
                          "split": "real_test" if i < 50 else "train", "weight": 1.0}) + "\n")
print("appended", len(data), "rows (50 real_test)")
