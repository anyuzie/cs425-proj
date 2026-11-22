"""Assemble extractor.jsonl (split by seed) and profiles.jsonl / dev_profiles.jsonl."""
import json, random
from collections import Counter
random.seed(425)
W, OUT = "/home/claude/data/work", "/home/claude/data/out"
T = ["R", "I", "A", "S", "E", "C"]
FULL = dict(zip(["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"], T))
def slots(**kw): return {t: kw.get(t, "neutral") for t in T}
L = lambda p: [json.loads(x) for x in open(p)]

# extractor
acts = {a["seed_id"]: a for a in L(f"{OUT}/activities.jsonl")}
# audit fixes: supervising cooks is management; personnel records are record-keeping
acts["IA_117"]["type"] = "E"; acts["IA_167"]["type"] = "C"
# blind-relabel adjudication: activities that genuinely span two types get both labels;
# the 24 O*NET Career Type rows (IA_001-IA_024) keep O*NET's own label
DUAL = {"IA_025": "RI", "IA_061": "RI", "IA_062": "RI", "IA_063": "RI", "IA_064": "RI",
        "IA_139": "SC", "IA_140": "SC", "IA_153": "EA", "IA_175": "EC", "IA_178": "EA",
        "IA_180": "EA", "IA_182": "EI", "IA_187": "ES", "IA_043": "RI", "IA_057": "RS"}
ONE = {"IA_120": "AC"}           # checking ingredient freshness is inspection, not creative cooking
for k, v in DUAL.items(): acts[k]["type"] = v
for k, v in ONE.items(): acts[k]["type"] = v
reph = {r["seed_id"]: r for r in json.load(open(f"{W}/reph_A.json")) + json.load(open(f"{W}/reph_B.json"))}
rows = []
for sid, a in acts.items():
    pos, neg = slots(**{t: "positive" for t in a["type"]}), slots(**{t: "negative" for t in a["type"]})
    rows.append({"text": a["activity"], "slots": pos, "reason": "enjoyment", "seed_id": sid, "kind": "seed"})
    for s in reph[sid]["reph"]:
        rows.append({"text": s, "slots": pos, "reason": "enjoyment", "seed_id": sid, "kind": "rephrase"})
    rows.append({"text": reph[sid]["neg"], "slots": neg, "reason": "enjoyment", "seed_id": sid, "kind": "negation"})
FIX = {"ENJ_014": slots(A="negative", C="positive"), "FAM_029": slots(E="positive")}
for r in L(f"{OUT}/synthetic_seeds.jsonl"):
    if r["seed_id"] in FIX: r["slots"] = FIX[r["seed_id"]]
    rows.append({**r, "kind": "seed"})
# extra seeds: negatives, forced-choice answers, familiarity, neutral/lukewarm
cnt = Counter()
for x in json.load(open(f"{W}/extra_A.json")) + json.load(open(f"{W}/extra_B.json")):
    if x["reason"] == "familiarity": pre = "FAM"
    elif x["reason"] == "neither": pre = "NEU"
    elif len(x["text"].split()) <= 12 and x in json.load(open(f"{W}/extra_A.json"))[115:]: pre = "ANS"
    else: pre = "NEG"
    base_n = {"FAM": 30, "NEU": 30}.get(pre, 0); cnt[pre] += 1
    sl = slots(**({x["type"]: x["polarity"]} if x["type"] else {}))
    rows.append({"text": x["text"], "slots": sl, "reason": x["reason"],
                 "seed_id": f"{pre}_{base_n + cnt[pre]:03d}", "kind": "seed"})

# extra Conventional seeds: 80 positive -> ENJ_031+, 20 negative -> NEG_116+
cp = cn = 0
for x in json.load(open(f"{W}/extra_C.json")):
    if x["polarity"] == "positive": cp += 1; sid = f"ENJ_{30 + cp:03d}"
    else: cn += 1; sid = f"NEG_{115 + cn:03d}"
    rows.append({"text": x["text"], "slots": slots(C=x["polarity"]), "reason": "enjoyment", "seed_id": sid, "kind": "seed"})

# critical-review additions
qa = json.load(open(f"{W}/qa_pairs.json")); xd = json.load(open(f"{W}/extra_D.json"))
for i, x in enumerate(qa["qa"], 1):     # replies that only make sense with the question: train on the pair
    rows.append({"text": x["answer"], "context": x["question"],
                 "slots": slots(**{x["type"]: x["polarity"] if x["polarity"] != "neutral" else "positive"}),
                 **({"strength": {x["type"]: "dontmind"}} if x["polarity"] == "neutral" else {}),
                 "reason": "enjoyment", "seed_id": f"QA_{i:03d}", "kind": "qa"})
for i, x in enumerate(qa["hard_pos"], 1):
    rows.append({"text": x["text"], "slots": slots(**{x["type"]: "positive"}), "reason": "enjoyment", "seed_id": f"HP_{i:03d}", "kind": "seed"})
for i, x in enumerate(xd["fam_neg"], 1):
    rows.append({"text": x["text"], "slots": slots(**{x["type"]: "negative"}), "reason": "familiarity", "seed_id": f"FAM_{72 + i:03d}", "kind": "seed"})
for i, x in enumerate(xd["lukewarm"], 1):
    rows.append({"text": x["text"], "slots": slots(**{x["type_mentioned"]: "positive"}), "strength": {x["type_mentioned"]: "dontmind"},
                 "reason": "enjoyment", "seed_id": f"NEU_{72 + i:03d}", "kind": "seed"})   # 'I don't mind X' = weak positive

gr = json.load(open(f"{W}/graded.json"))
NM2L = {"Realistic": "R", "Investigative": "I", "Artistic": "A", "Social": "S", "Enterprising": "E", "Conventional": "C"}
POL = {"pos": "positive", "neg": "negative", "positive": "positive", "negative": "negative"}
for i, x in enumerate(gr["mild"], 1):
    tl = NM2L.get(x["type"], x["type"]); pol = POL[x["polarity"]]          # accept full type names as well as letters
    rows.append({"text": x["text"], "slots": slots(**{tl: pol}), "strength": {tl: "mild"},
                 "reason": "enjoyment", "seed_id": f"MILD_{i:03d}", "kind": "seed"})
# strong S/A dislikes, long multi-clause sentences, lexicon-coverage sentences
for k, x in enumerate(json.load(open(f"{W}/extra_E.json")), 1):
    pre = "XDS" if k <= 30 else "LONG" if k <= 90 else "COV"
    rows.append({"text": x["text"], "slots": slots(**{t: v[0] for t, v in x["labels"].items()}),
                 "strength": {t: v[1] for t, v in x["labels"].items()}, "reason": "enjoyment",
                 "seed_id": f"{pre}_{k:03d}", "kind": "seed"})
import re as _re
# indifference ("neutral", "no strong preference", "just okay") = don't mind (+0.2); only uncertainty is no stance
UNSURE = _re.compile(r"not sure|depends|dunno|\bidk\b|no idea|hard to say|unsure|never thought", _re.I)
for r_ in rows:
    if r_.get("strength") and "dontmind" in r_["strength"].values() and UNSURE.search(r_["text"]):
        r_["slots"] = slots(); r_["strength"] = {}; r_["reason"] = "neither"
# strength of every stance (blind-annotated); soft training targets make confidence track strength
SM = json.load(open(f"{W}/strength_map.json"))
for r in rows:
    if "strength" not in r:
        got = SM.get(r.get("context", "") + " || " + r["text"], {})
        r["strength"] = {t: got.get(t, "strong") for t in T if r["slots"][t] != "neutral"}

# second, rule-guided strength pass overrides the first (consistent strong/mild boundary)
SM2 = json.load(open(f"{W}/strength_map_v2.json"))
for r in rows:
    got = SM2.get(r.get("context", "") + " || " + r["text"])
    if got:
        for t, v in got.items():
            if t in r["strength"] and r["strength"][t] != "dontmind": r["strength"][t] = v

# third-vote adjudication in the presence/strength scheme (majority of data + 2 blind readers).
# Two-type activities stay two-type, and O*NET's own Career Type labels (IA_001-IA_024) stay.
LV = {1.0: ("positive", "strong"), 0.55: ("positive", "mild"), 0.2: ("positive", "dontmind"),
      -0.55: ("negative", "mild"), -1.0: ("negative", "strong")}
OV = json.load(open(f"{W}/liking_override.json"))
for r in rows:
    new = OV.get(r.get("context", "") + " || " + r["text"])
    if new is None: continue
    old = {t for t in T if r["slots"][t] != "neutral"}
    if r["seed_id"].startswith("IA_") and len(old) == 2 and set(new) < old: continue        # keep dual labels
    if r["seed_id"].startswith("IA_") and int(r["seed_id"][3:6]) <= 24 and set(new) != old: continue   # keep O*NET's type
    if r["seed_id"].startswith("IA_") and set(new) - old and not old <= set(new): continue  # no type swaps on activities
    r["slots"] = slots(**{t: LV[float(v)][0] for t, v in new.items()})
    r["strength"] = {t: LV[float(v)][1] for t, v in new.items()}
    r["reason"] = r["reason"] if new else "neither"
    if new and r["reason"] == "neither": r["reason"] = "enjoyment"

# split by seed_id, 70/15/15, stratified by seed family + type
def strat(sid):
    r = next(x for x in rows if x["seed_id"] == sid)
    fam = sid.split("_")[0]
    ty = next((t for t in T if r["slots"][t] != "neutral"), "N")
    return fam + ty
by = {}
for sid in sorted({r["seed_id"] for r in rows}): by.setdefault(strat(sid), []).append(sid)
split = {}
for k, ids in by.items():
    random.shuffle(ids); n = len(ids)
    nt, nv = round(n * .15), round(n * .15)
    for i, sid in enumerate(ids):
        split[sid] = "test" if i < nt else "val" if i < nt + nv else "train"
for r in rows: r["split"] = split[r["seed_id"]]

# template frames on a 250-sentence sample (stays in its seed's split)
def lo(s): return s if (s[:2] in ("I ", "I'") or s[:3]=="I'") else s[0].lower() + s[1:]
FR = [lambda s: "Honestly, " + lo(s), lambda s: "I think " + lo(s),
      lambda s: "I guess " + lo(s), lambda s: s.rstrip(".!") + ", at least most of the time.",
      lambda s: "To be fair, " + lo(s)]
# frames: any frame on enjoyment rows; only natural prefixes on familiarity/neutral seeds (1 each)
base = [r for r in rows if r["kind"] in ("rephrase", "negation") or r["seed_id"][:3] in ("ENJ", "NEG", "ANS")]
SAFE_FAM = [lambda s: "Honestly, " + lo(s), lambda s: "To be fair, " + lo(s), lambda s: "Well, " + lo(s)]
SAFE_NEU = [lambda s: "Honestly, " + lo(s), lambda s: "Hmm, " + lo(s), lambda s: "Um, " + lo(s)]
fn = [r for r in rows if r["seed_id"][:3] in ("FAM", "NEU")]
for r in fn:
    fr = random.choice(SAFE_FAM if r["reason"] == "familiarity" else SAFE_NEU)
    rows.append({**r, "text": fr(r["text"]), "kind": "frame"})
for r in random.sample(base, 250 - len(fn)):
    # hand-written seeds only get the two openers that read naturally on any sentence
    pool = FR if r["seed_id"][:2] == "IA" else [FR[0], FR[4]]
    rows.append({**r, "text": random.choice(pool)(r["text"]), "kind": "frame"})

# 160 two-type combos, built inside each split from different-type single-slot sentences
singles = [r for r in rows if r["kind"] in ("rephrase", "negation") or (r["seed_id"][:3] in ("ENJ", "NEG") and r["kind"] == "seed")]
JOIN = [" but ", ", and ", ". On the other hand, ", ", though "]
target = {"train": 112, "val": 24, "test": 24}
for sp, n in target.items():
    pool = [r for r in singles if r["split"] == sp]; made, seen = 0, set()
    while made < n:
        a, b = random.sample(pool, 2)
        ta = {t for t in T if a["slots"][t] != "neutral"}; tb = {t for t in T if b["slots"][t] != "neutral"}
        if ta & tb or (a["seed_id"], b["seed_id"]) in seen: continue
        seen.add((a["seed_id"], b["seed_id"]))
        same = {a["slots"][t] for t in ta} == {b["slots"][t] for t in tb}
        j = random.choice([", and ", " and "] if same else JOIN); bt = b["text"]   # contrast words only when signs differ
        if not j.startswith("."): bt = lo(bt)
        s = slots(**{t: a["slots"][t] for t in ta}, **{t: b["slots"][t] for t in tb})
        stg = {**{t: a["strength"][t] for t in ta}, **{t: b["strength"][t] for t in tb}}
        rows.append({"text": a["text"].rstrip(".!") + j + bt, "slots": s, "reason": "enjoyment",
                     "seed_id": a["seed_id"] + "+" + b["seed_id"], "kind": "combo", "split": sp, "strength": stg})
        made += 1

# bare O*NET imperatives: half weight, and kept out of val/test so they do not inflate F1
rows = [r for r in rows if not (r["kind"] == "seed" and r["seed_id"].startswith("IA_") and "+" not in r["seed_id"] and r["split"] != "train")]
for r in rows:
    r["weight"] = 0.5 if (r["kind"] == "seed" and r["seed_id"].startswith("IA_") and "+" not in r["seed_id"]) else 1.0
    r.setdefault("context", "")
# labels: direction + strength
#   slots    : per type positive / neutral / negative  (3-class head; confidence = softmax prob of the chosen class)
#   strength : per non-neutral type, how much, 0..1    (sigmoid head): strong 1.0, mild 0.55, don't mind 0.2
SV = {"strong": 1.0, "mild": 0.55, "dontmind": 0.2}
for r in rows:
    r["strength_label"] = r.pop("strength")
    r["strength"] = {t: SV[v] for t, v in r["strength_label"].items()}
    for k in ("soft", "present", "liking"): r.pop(k, None)
import re
_seen, _rows = set(), []
for r in rows:                                   # drop exact duplicates (after normalising case/punctuation)
    k = re.sub(r"[^a-z ]", "", (r.get("context", "") + " || " + r["text"]).lower()).strip()
    if k not in _seen: _seen.add(k); _rows.append(r)
rows = _rows
with open(f"{OUT}/extractor.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("extractor rows", len(rows), Counter(r["kind"] for r in rows), Counter(r["split"] for r in rows),
      Counter(r["reason"] for r in rows))
print("seeds per split", Counter(split.values()))
pc = Counter((t, r["slots"][t]) for r in rows for t in T if r["slots"][t] != "neutral")
print("non-neutral labels", dict(sorted(pc.items())))

# profiles are built by build_profiles_v3.py
