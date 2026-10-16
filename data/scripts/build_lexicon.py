"""RIASEC term lexicon derived from O*NET (not hand-written).
For each type, compare words in the task statements of occupations dominant in that type (by z) against all other
task statements, using the weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008).
Terms with z > 2 for a type are kept, top 60 per type. Also writes a small hand-made intensity/negation list."""
import pandas as pd, numpy as np, re, json, collections
O = "/home/claude/data/out"; T = ["R", "I", "A", "S", "E", "C"]
STOP = set("""a an the and or of to in on for with by as at from into that this these those be is are was were been being it its their
them they he she his her you your we our i me my not no can may will such other others using use used including include includes
as well per via any all each both more most than then so if when where while which who whom whose what about over under up down out
also into through during before after between within without against among upon""".split())
occ = pd.read_csv(f"{O}/occupations.csv"); occ = occ[occ.has_riasec]
dom = occ.set_index("code")[[f"z_{t}" for t in T]].idxmax(axis=1).str[-1]
tasks = pd.read_csv(f"{O}/tasks.csv"); tasks = tasks[tasks.code.isin(dom.index)]
tok = lambda s: [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2]
counts = {t: collections.Counter() for t in T}
for c, txt in zip(tasks.code, tasks.Task): counts[dom[c]].update(tok(txt))
allc = collections.Counter(); [allc.update(v) for v in counts.values()]
a0 = sum(allc.values()); alpha = {w: 0.01 * n for w, n in allc.items()}; A0 = sum(alpha.values())
lex = {}
for t in T:
    yi = counts[t]; ni = sum(yi.values())
    yj = allc - yi; nj = sum(yj.values())
    sc = {}
    for w, n in allc.items():
        if n < 15: continue
        a = alpha[w]; i_, j_ = yi[w], yj[w]
        d = np.log((i_ + a) / (ni + A0 - i_ - a)) - np.log((j_ + a) / (nj + A0 - j_ - a))
        var = 1 / (i_ + a) + 1 / (j_ + a)
        sc[w] = d / np.sqrt(var)
    top = sorted(((w, s) for w, s in sc.items() if s > 2), key=lambda x: -x[1])[:60]
    lex[t] = [{"term": w, "z": round(float(s), 2)} for w, s in top]
INTENSITY = {
 "strong_positive": ["love", "loves", "loved", "passionate", "obsessed", "favourite", "favorite", "best part", "dream", "definitely",
                     "absolutely", "really enjoy", "so fun", "can do it for hours", "shiok", "thrilling", "amazing", "addicted"],
 "mild_positive": ["kinda", "kind of", "sort of", "i guess", "probably", "slightly", "a bit", "somewhat", "not bad", "alright",
                   "okay", "ok", "wouldn't mind", "sounds fine", "maybe", "i suppose", "lean towards"],
 "strong_negative": ["hate", "hated", "can't stand", "cannot stand", "nightmare", "torture", "never", "no way", "absolutely not",
                     "dread", "drives me crazy", "couldn't pay me", "worst", "awful", "pass"],
 "mild_negative": ["not really", "not my thing", "not for me", "meh", "a bit boring", "rather not", "not keen", "not into",
                   "probably not", "less into", "sian"],
 "dontmind": ["either", "don't mind", "dont mind", "both fine", "either also can", "no preference"],
 "unsure": ["not sure", "depends", "dunno", "idk", "no idea", "hard to say"],
 "negators": ["not", "no", "never", "don't", "dont", "doesn't", "isn't", "wouldn't", "can't", "hardly"],
 "familiarity": ["my course", "my major", "my diploma", "poly", "my parents", "my dad", "my mum", "my family", "internship",
                 "part-time", "grew up", "everyone says", "people say", "safe choice", "what i know", "used to it"],
}
json.dump({"source": "O*NET 31.0 task statements; weighted log-odds with informative prior (Monroe et al., 2008)",
           "types": lex, "intensity": INTENSITY}, open(f"{O}/riasec_lexicon.json", "w"), indent=1)
for t in T: print(t, ", ".join(x["term"] for x in lex[t][:18]))
# coverage: how many of each type's terms appear anywhere in the training sentences
r = [json.loads(l) for l in open(f"{O}/extractor.jsonl")]
words = collections.Counter(w for x in r for w in tok(x["text"]))
gaps = {}
for t in T:
    terms = [x["term"] for x in lex[t]]; miss = [w for w in terms if words[w] == 0]
    gaps[t] = miss; print(f"{t}: {len(terms) - len(miss)}/{len(terms)} lexicon terms appear in training; missing e.g. {miss[:10]}")
json.dump(gaps, open("/home/claude/data/work/lexicon_gaps.json", "w"), indent=1)
