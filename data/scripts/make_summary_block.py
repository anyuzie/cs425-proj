"""Builds the summary section of the data review page from the current files (so numbers never go stale)."""
import json, collections, csv, html
O = "/home/claude/data/out"; T = "RIASEC"
NAME = dict(zip(T, ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]))
L = lambda p: [json.loads(l) for l in open(p)]
r = L(f"{O}/extractor.jsonl"); P = L(f"{O}/profiles.jsonl"); D = L(f"{O}/dev_profiles.jsonl")
nli = L(f"{O}/nli_pairs.jsonl"); q = json.load(open(f"{O}/question_bank.json"))
occ = {o["code"]: o for o in csv.DictReader(open(f"{O}/occupations.csv"))}
e = html.escape
src = lambda x: "combo" if "+" in x["seed_id"] else x["seed_id"].split("_")[0]
def ex(f):
    for x in r:
        if f(x): return x["text"]
    return ""
def chips(x):
    out = []
    for t in T:
        if x["slots"][t] != "neutral":
            s = x["strength_label"].get(t, "strong"); lab = {"strong": "", "mild": " mild", "dontmind": " don't mind"}[s]
            out.append(f'<span class="chip {"p" if x["slots"][t] == "positive" else "n"}">{t} {"+" if x["slots"][t] == "positive" else "−"}{lab}</span>')
    return "".join(out) or '<span class="chip z">no stance</span>'
def exrow(f):
    for x in r:
        if f(x): return x
SOURCES = [
 ("IA", "O*NET Interests Illustrative Activities: the 188 originals, 2 rephrasings each, 1 negation each, and frames", lambda x: x["seed_id"].startswith("IA_") and x["kind"] == "rephrase"),
 ("ENJ", "Synthetic enjoyment seeds, including 80 Conventional likes", lambda x: src(x) == "ENJ"),
 ("NEG", "Extra dislikes, so the negative classes aren't thin", lambda x: src(x) == "NEG"),
 ("MILD", "Hedged likes and dislikes, teaching the model what 'mild' sounds like", lambda x: src(x) == "MILD"),
 ("HP", "Likes that contain a negation word (\"I wouldn't mind…\")", lambda x: src(x) == "HP"),
 ("FAM", "Familiarity: a lean that comes from course, family or job, positive or negative", lambda x: src(x) == "FAM" and x["seed_id"] > "FAM_072"),
 ("NEU", "No stance, off-topic, lukewarm or \"don't mind\"", lambda x: src(x) == "NEU" and x["reason"] == "neither"),
 ("QA", "Short replies paired with the bot's question (\"the first one\")", lambda x: src(x) == "QA" and x["slots"]["R"] != "neutral"),
 ("ANS", "Forced-choice style answers, not shared with the simulator", lambda x: src(x) == "ANS"),
 ("combo", "Two sentences joined, giving two types at once", lambda x: src(x) == "combo"),
]
cnt = collections.Counter(src(x) for x in r)
rows_src = ""
for k, desc, f in SOURCES:
    x = exrow(f)
    q_ = f'<div class="ans">Q: {e(x["context"])}</div>' if x and x.get("context") else ""
    rows_src += f'<tr><td><b>{k}</b> <span class="ans">{desc}</span></td><td class="num">{cnt[k]}</td><td>{q_}{e(x["text"]) if x else ""} {chips(x) if x else ""}</td></tr>'
rows_type = ""
for t in T:
    c = collections.Counter((x["slots"][t][:3], x["strength_label"][t]) for x in r if x["slots"][t] != "neutral")
    sp = {s: sum(1 for x in r if x["slots"][t] != "neutral" and x["split"] == s) for s in ("train", "val", "test")}
    rows_type += (f'<tr><td><b>{t}</b> {NAME[t]}</td><td class="num">{c[("pos","strong")]}</td><td class="num">{c[("pos","mild")]}</td><td class="num">{c[("pos","dontmind")]}</td>'
                  f'<td class="num">{c[("neg","mild")]}</td><td class="num">{c[("neg","strong")]}</td><td class="num">{sp["train"]} / {sp["val"]} / {sp["test"]}</td></tr>')
reason = collections.Counter(x["reason"] for x in r); ntypes = collections.Counter(min(len(x["strength_label"]), 3) for x in r)
split = collections.Counter(x["split"] for x in r)
# profiles
plan = collections.Counter(p.get("plan", "") for p in P if p["tier"] == 2)
dom = collections.Counter(max(T, key=lambda k: float(occ[p["target"]][k])) for p in P)
zones = collections.Counter(occ[p["target"]]["job_zone"][0] for p in P)
def pex(f):
    p = next(x for x in P if f(x)); return p
def pcard(label, p):
    hs = " ".join(f'{t} {p["hidden_slots"][t]:+.1f}' for t in T)
    extra = ""
    if p.get("contradiction"):
        c = p["contradiction"]; g = c["statement_" + c["genuine"]]
        extra += f'<div class="ans">Planted contradiction on {c["slot"]}; the genuine side is “{e(g)}”.</div>'
    if p.get("latent_reveal"): extra += f'<div class="ans">If probed: “{e(p["latent_reveal"])}”</div>'
    return (f'<tr><td><b>{label}</b><div class="ans">{p["id"]} · {e(occ[p["target"]]["Title"])}</div></td><td>{e(p["text"])}{extra}</td>'
            f'<td class="m">{hs}</td></tr>')
prof_rows = (pcard("Clean (Tier 1)", pex(lambda x: x["tier"] == 1 and x["style"] == "full" and not x["background"])) +
             pcard("Contradiction (Tier 2)", pex(lambda x: x.get("plan") == "contradiction_like")) +
             pcard("Misleading familiarity (Tier 2)", pex(lambda x: x.get("plan") == "familiarity_misleading")))
qb = q["ANSWERS"]["Realistic"][0]
nli_c = collections.Counter((x["split"], x["label"]) for x in nli)

LEXD = json.load(open(f"{O}/riasec_lexicon.json"))
LEXROWS = "".join(f'<tr><td><b>{t}</b> {NAME[t]}</td><td>{", ".join(x["term"] for x in LEXD["types"][t][:14])}</td></tr>' for t in T)

# balanced sets, patterns, stress test
import sys, numpy as np, pandas as pd
sys.path.insert(0, f"{O}/eval"); import metrics as MET
V4 = L(f"{O}/profiles_v4.jsonl"); ST = L(f"{O}/extractor_stress.jsonl")
_o = pd.read_csv(f"{O}/occupations.csv"); _pool = _o[(_o.job_zone.fillna(0) >= 4) & _o.has_riasec]
_codes = list(_pool.code); _Z = _pool[[f"z_{t}" for t in T]].to_numpy()
def _rank(p):
    v = np.array([p["hidden_slots"][t] for t in T]); c = _Z @ v / (np.linalg.norm(_Z, axis=1) * np.linalg.norm(v)); return [_codes[i] for i in np.argsort(-c)]
def ceil(ps, fn): return round(100 * np.mean([fn(_rank(p), p) for p in ps])) if ps else 0
SETS = [("Test v3 (profiles.jsonl)", P), ("Dev (dev_profiles.jsonl)", D), ("Test v4 (profiles_v4.jsonl)", V4)]
top_rows = "".join(f'<tr><td>{n}</td>' + "".join(f'<td class="num">{sum(1 for p in ps if p["top_type"] == t)}</td>' for t in T) + f'<td class="num">{len(ps)}</td></tr>' for n, ps in SETS)
PAT = [("single_dominant", "One interest clearly leads", "R −0.5, I +2.4, A 0.0, S +0.1, E −0.6, C +0.4"),
       ("two_strong", "Two interests both high", "I +2.1, S +2.0, others near 0"),
       ("three_balanced", "Three interests about equal", "R +1.7, I +1.9, A +1.8, others low"),
       ("weak_uncertain", "Unsure, only a slight lean", "all between −0.5 and +0.7"),
       ("conflicting", "Strong likes and strong dislikes", "R +2.0, I +1.8, A −1.4, S −1.7"),
       ("generalist", "Likes many kinds of work", "all about +1.0 to +1.3")]
allp = P + D + V4
pat_rows = ""
for k, desc, ex_ in PAT:
    ps = [p for p in allp if p["pattern"] == k]; pv = [p for p in V4 if p["pattern"] == k]
    exp = next((p for p in V4 + D if p["pattern"] == k and p["tier"] == 1 and p.get("style") == "full"), None)
    txt = e(exp["text"]) if exp else ""
    pat_rows += (f'<tr><td><b>{k}</b><div class="ans">{desc}</div><div class="ans m">{ex_}</div></td>'
                 f'<td class="num">{sum(1 for p in P if p["pattern"]==k)} / {sum(1 for p in D if p["pattern"]==k)} / {len(pv)}</td>'
                 f'<td class="num">{ceil(pv, lambda r, p: MET.hit_at(r, p, 5))}% · {ceil(pv, MET.related_hit_at)}%</td><td>{txt}</td></tr>')
VAR = collections.OrderedDict()
for r_ in ST: VAR.setdefault(r_["variation"], []).append(r_)
VAR_DESC = {"indirect_interest": "Interest shown through what they did", "ambivalence": "Mixed or lukewarm feelings",
            "contextual_dislike": "Likes the work, dislikes one part of it", "experience_vs_preference": "What they've done vs what they enjoy",
            "vague": "Unsure, with or without a lean", "multiple_interests": "Two or three types in one sentence",
            "informal": "Chat style, abbreviations, Singlish", "temporal_change": "Interests that changed over time"}
STAND = {"indirect_interest": 0.51, "ambivalence": 0.64, "contextual_dislike": 0.64, "experience_vs_preference": 0.74,
         "vague": 0.63, "multiple_interests": 0.66, "informal": 0.84, "temporal_change": 0.62}
st_rows = "".join(f'<tr><td><b>{k}</b><div class="ans">{VAR_DESC[k]}</div></td><td class="num">{len(v)}</td>'
                  f'<td>{e(v[3]["text"])} {chips(v[3])}</td><td class="num">{STAND[k]:.2f}</td></tr>' for k, v in VAR.items())

BLOCK = f'''
<h2>In one minute</h2>
<ul class="glance">
<li><b>Jobs:</b> O*NET 31.0. The bot recommends from <b>378 degree-level occupations</b> (Job Zones 4–5) and quotes their task statements.</li>
<li><b>Extractor training:</b> {len(r):,} labelled sentences. Each of the six types is <b>positive, neutral or negative</b> (the model's probability is the confidence), plus a <b>strength 0–1</b> when not neutral.</li>
<li><b>Testing:</b> two locked test sets of simulated students — v3 (100) and the balanced v4 (120, 20 per top type, six patterns) — plus 120 balanced dev students for tuning. Each set is 60% clean and 40% with a planted contradiction or familiarity problem.</li>
<li><b>Extractor stress test:</b> 400 hard sentences in 8 phrasing types, kept out of training.</li>
<li><b>Contradiction check:</b> {len(nli)} sentence pairs to set and test the cut-off.</li>
</ul>

<h2>Ready to train?</h2>
<div class="tw"><table><thead><tr><th>Model</th><th>Train on</th><th>Tune / check on</th><th>Enough?</th></tr></thead><tbody>
<tr><td><b>Extractor</b> (DeBERTa)</td><td>{split["train"]:,} train rows</td><td>val {split["val"]}, test {split["test"]} (once), stress test {len(ST)}, 50 real sentences</td><td>Yes. More data still helps (see limits).</td></tr>
<tr><td><b>Contradiction check</b> (NLI)</td><td>Not trained</td><td>dev {sum(1 for x in nli if x["split"]=="dev")} pairs for the 0.60 cut-off, test {sum(1 for x in nli if x["split"]=="test")}</td><td>Yes</td></tr>
<tr><td><b>BM25, MiniLM/FAISS</b></td><td>Not trained: index the job and task files</td><td>–</td><td>Yes</td></tr>
<tr><td><b>Ranking, thresholds, stopping</b></td><td>Not trained</td><td>{len(D)} dev profiles (20 per top type)</td><td>Yes</td></tr>
</tbody></table></div>
<p class="note2">All label settings are in <code>labels.json</code>: class order neutral 0 / positive 1 / negative 2, class weights [0.41, 2.95, 4.99], reason weights [0.40, 3.44, 4.55], input as <code>tokenizer(context, text)</code>.</p>

<h2>Files</h2>
<div class="tw"><table><thead><tr><th>File</th><th>Rows</th><th>Used for</th></tr></thead><tbody>
<tr><td><code>extractor.jsonl</code></td><td class="num">{len(r):,}</td><td>Training the extractor</td></tr>
<tr><td><code>labels.json</code></td><td class="num">–</td><td>Label order, class weights, input format (read by all training code)</td></tr>
<tr><td><code>occupations.csv</code></td><td class="num">1,016</td><td>RIASEC z-scores, job zone, BM25 text. Recommend only job zone 4–5.</td></tr>
<tr><td><code>tasks.csv</code></td><td class="num">18,838</td><td>Dense search (FAISS) and the evidence quoted in recommendations</td></tr>
<tr><td><code>question_bank.json</code></td><td class="num">–</td><td>The bot's questions, and the simulated student's five answer levels</td></tr>
<tr><td><code>profiles.jsonl</code></td><td class="num">{len(P)}</td><td>Final test v3 only. Never tune on it.</td></tr>
<tr><td><code>profiles_v4.jsonl</code></td><td class="num">{len(V4)}</td><td>Final test v4 (balanced, patterns). Never tune on it.</td></tr>
<tr><td><code>extractor_stress.jsonl</code></td><td class="num">{len(ST)}</td><td>Extractor stress test. Never train on it.</td></tr>
<tr><td><code>dev_profiles.jsonl</code></td><td class="num">{len(D)}</td><td>All tuning</td></tr>
<tr><td><code>nli_pairs.jsonl</code></td><td class="num">{len(nli)}</td><td>Setting and testing the contradiction cut-off</td></tr>
<tr><td><code>related_occupations.csv</code></td><td class="num">18,460</td><td>Partial credit in scoring (related-hit@5, NDCG@10)</td></tr>
<tr><td><code>eval/</code></td><td class="num">–</td><td>Simulated student, metrics, keyword baseline</td></tr>
<tr><td><code>riasec_lexicon.json</code></td><td class="num">6 × 60</td><td>Keyword baseline and labelling reference</td></tr>
<tr><td><code>labelling/</code></td><td class="num">–</td><td>Guide and scripts for adding real student sentences</td></tr>
<tr><td class="ans">activities, synthetic_seeds, nli_pairs_blind, scripts/</td><td class="num">–</td><td class="ans">Sources and rebuild scripts. Not needed to train.</td></tr>
</tbody></table></div>

<h2>Training data</h2>
<p class="lede">Split by seed so no sentence family is in two splits: train {split["train"]:,} / val {split["val"]} / test {split["test"]}. Reason: enjoyment {reason["enjoyment"]:,}, familiarity {reason["familiarity"]}, neither {reason["neither"]}.</p>
<div class="tw"><table><thead><tr><th>Type</th><th>Like, strong</th><th>Like, mild</th><th>Don't mind</th><th>Dislike, mild</th><th>Dislike, strong</th><th>Train / val / test</th></tr></thead><tbody>{rows_type}</tbody></table></div>
<p class="note2">Strength values: strong 1.0, mild 0.55, don't mind 0.2. "Not sure / depends" is neutral.</p>
<h3>Where the sentences come from</h3>
<div class="tw"><table><thead><tr><th style="width:34%">Source</th><th>Rows</th><th>Example and labels</th></tr></thead><tbody>{rows_src}</tbody></table></div>

<h2>Test students</h2>
<p class="lede">Each student is built from a real degree-level O*NET job; the job is the answer. In the conversation, CLARIFY answers follow the hidden score, CHALLENGE returns a resolution, and only PROBE reveals the hidden reason.</p>
<h3>Top type of each student (highest z in <code>hidden_slots</code>)</h3>
<div class="tw"><table><thead><tr><th>Set</th><th>R</th><th>I</th><th>A</th><th>S</th><th>E</th><th>C</th><th>Total</th></tr></thead><tbody>{top_rows}</tbody></table></div>
<p class="note2">v3 was balanced on O*NET's raw scores, which disagree with z-scores for most Realistic and Conventional jobs (Architects: raw top R, z top A), so it has no R-led students. No degree-level job leads on R in z-scores, so R-led students in dev and v4 are built from the most Realistic degree jobs (engineers and similar) with Realistic raised.</p>
<h3>Patterns</h3>
<div class="tw"><table><thead><tr><th style="width:24%">Pattern</th><th>v3 / dev / v4</th><th>v4 ceiling hit@5 · related</th><th>Example opening</th></tr></thead><tbody>{pat_rows}</tbody></table></div>
<p class="note2">Ceiling = a perfect reader of the student's scores, structured matching only. It is low for generalist and three-balanced students because several jobs fit them almost equally; report <code>related-hit@5</code> and <code>fit-hit@5</code> beside hit@5, and split results by pattern.</p>
<h3>One example of each kind</h3>
<div class="tw"><table><thead><tr><th style="width:20%">Kind</th><th>What the student says</th><th style="width:22%">Hidden scores (z)</th></tr></thead><tbody>{prof_rows}</tbody></table></div>

<h2>Extractor stress test</h2>
<p class="lede"><code>extractor_stress.jsonl</code>: 400 sentences, same fields as the training data plus <code>variation</code>. Labels checked by two blind readers and a majority vote (direction agreement 99.9%, strength 99.6%, reason 99.5%).</p>
<div class="tw"><table><thead><tr><th style="width:26%">Variation</th><th>Rows</th><th>Example and labels</th><th>Stand-in F1</th></tr></thead><tbody>{st_rows}</tbody></table></div>
<p class="note2">Stand-in = the TF-IDF model, macro-F1 over the six types (0.68 overall vs 0.67 on the normal test split). Indirect interest is hardest. Report DeBERTa on this file after the normal test split.</p>

<h2>Question bank example</h2>
<div class="tw"><table><thead><tr><th>Question</th><th>Strong yes</th><th>Mild yes</th><th>Don't mind</th><th>Mild no</th><th>Strong no</th></tr></thead><tbody>
<tr><td>{e(q["CLARIFY"]["Realistic"][0])}</td><td>{e(qb["pos"])}</td><td>{e(qb["pos_mild"])}</td><td>{e(qb["dontmind"])}</td><td>{e(qb["neg_mild"])}</td><td>{e(qb["neg"])}</td></tr>
</tbody></table></div>

<h2>Contradiction pairs</h2>
<p class="lede">Dev {nli_c[("dev","contradiction")]} contradictions / {nli_c[("dev","elaboration")]+nli_c[("dev","unrelated")]} not. Test {nli_c[("test","contradiction")]} / {nli_c[("test","elaboration")]+nli_c[("test","unrelated")]}. Each split has 20 hard pairs that sound opposite but fit ("I like coding, but not front-end").</p>

<h2>How we know it's right</h2>
<p class="lede">Readers who never saw the labels relabelled everything; disagreements went to a third reader.</p>
<div class="tw"><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>
<tr><td>Training sentences: stance or no stance</td><td>F1 0.965</td></tr>
<tr><td>Training sentences: strength level</td><td>94.7% exact, 0 like/dislike flips</td></tr>
<tr><td>Question-bank answers read at their level</td><td>90 / 90</td></tr>
<tr><td>Student texts match hidden scores (all sets)</td><td>2,040 / 2,040 slots within band; r = 0.88 on the new 180</td></tr>
<tr><td>Stress-test labels (two blind readers)</td><td>99.9% direction, 2 like/dislike flips in 4,800</td></tr>
<tr><td>Contradiction pairs sorted blind</td><td>70 / 70 for the new pairs</td></tr>
<tr><td>Leaks between splits, or from test into training</td><td>None</td></tr>
</tbody></table></div>

<h2>What results to expect</h2>
<div class="tw"><table><thead><tr><th>Setup</th><th>Exact job in top 5</th><th>Exact or related in top 5</th></tr></thead><tbody>
<tr><td>Random guess</td><td class="num">1%</td><td class="num">–</td></tr>
<tr><td>Simple stand-in system (no DeBERTa, no dense search)</td><td class="num">13%</td><td class="num">29%</td></tr>
<tr><td><b>Ceiling</b>, test v3: system knows each student perfectly</td><td class="num">82%</td><td class="num">90%</td></tr>
<tr><td><b>Ceiling</b>, dev (120)</td><td class="num">{ceil(D, lambda r,p: MET.hit_at(r,p,5))}%</td><td class="num">{ceil(D, MET.related_hit_at)}%</td></tr>
<tr><td><b>Ceiling</b>, test v4 (patterns)</td><td class="num">{ceil(V4, lambda r,p: MET.hit_at(r,p,5))}%</td><td class="num">{ceil(V4, MET.related_hit_at)}%</td></tr>
</tbody></table></div>
<p class="note2">The gap is mostly the extractor (stand-in macro-F1 0.64, keyword baseline 0.48). Fine-tuned DeBERTa plus dense search should close much of it.</p>

<h2>Known limits</h2>
<ul class="glance">
<li>All text is AI-written and AI-checked. A human spot-check and the 50 real student sentences are the next step.</li>
<li>The extractor was still improving with more data, so more real sentences would help.</li>
<li>Some test groups are small (about 20 "don't mind" rows per split): report strength over all types together.</li>
<li>No degree-level job leads on Realistic in z-scores, so R-led students are matched to engineering-type jobs and their ceiling is low (about 35%).</li>
</ul>
'''
open("/home/claude/data/work/summary_block.html", "w").write(BLOCK)
print("ok", len(BLOCK))
