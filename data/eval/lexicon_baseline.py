"""Keyword baseline for the extractor (no training). Report it next to DeBERTa:
"a keyword model using O*NET's own vocabulary gets X macro-F1; the trained extractor gets Y".
Rule: a type is mentioned if the sentence contains one of its O*NET lexicon terms; direction comes from sentiment cue
words (flipped by a negator just before the cue); strength comes from the intensity lists."""
import json, re, sys, os
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..")
LEX = json.load(open(os.path.join(D, "riasec_lexicon.json"))); T = "RIASEC"
TERMS = {t: {x["term"] for x in LEX["types"][t][:60]} for t in T}
I = LEX["intensity"]
POS = ["love", "enjoy", "like", "fun", "exciting", "interesting", "satisfying", "rewarding", "cool", "happy", "shiok", "great",
       "amazing", "favourite", "favorite", "passionate", "keen", "into", "prefer", "would rather", "sounds good", "fulfilling"]
NEG = ["hate", "dislike", "boring", "dull", "tedious", "sian", "nightmare", "torture", "awful", "drain", "can't stand", "meh",
       "not my thing", "not for me", "stressful", "uncomfortable", "rather not", "pass", "miserable"]
NEGATORS = set(I["negators"])
words = lambda s: re.findall(r"[a-z']+", s.lower())

def read(text):
    w = words(text); low = " ".join(w); out = {}
    stem = lambda x: x[:-1] if x.endswith("s") else x
    for t in T:
        if not any(stem(x) in {stem(y) for y in TERMS[t]} for x in w): continue
        pos = sum(low.count(c) for c in POS); neg = sum(low.count(c) for c in NEG)
        for i, x in enumerate(w):                       # "don't enjoy" flips a positive cue
            if x in NEGATORS and any(" ".join(w[i + 1:i + 3]).startswith(c) for c in POS): pos -= 1; neg += 1
        if pos == neg == 0: continue
        out[t] = "positive" if pos > neg else "negative"
    return out

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(os.path.join(D, "extractor.jsonl"))]
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    te = [r for r in rows if r["split"] == split]
    pred = [read(r["text"]) for r in te]
    f1 = {t: round(f1_score([r["slots"][t] for r in te], [p.get(t, "neutral") for p in pred], average="macro"), 3) for t in T}
    print(f"keyword baseline, {split} split, macro-F1 per type:", f1, "mean", round(sum(f1.values()) / 6, 3))
