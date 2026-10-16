"""Compare blind ratings (-2..+2, rater never saw numbers) to the O*NET z bands each text was written to."""
import json, sys
RAT = sys.argv[1] if len(sys.argv) > 1 else "work/faith_ratings.json"
r = json.load(open(RAT))
F = {"Realistic":"R","Investigative":"I","Artistic":"A","Social":"S","Enterprising":"E","Conventional":"C"}
P = {}
for f in ["out/profiles.jsonl", "out/dev_profiles.jsonl"]:
    for l in open(f): x = json.loads(l); P[x["id"]] = x
def band(z):
    if z >= 1.0: return "strong+"
    if z >= 0.3: return "mild+"
    if z > -0.4: return "neutral"
    if z > -0.8: return "mild-"
    return "strong-"
# what a faithful text must show, as an allowed range of the blind rating
OK = {"strong+": (1, 2), "mild+": (0, 2), "neutral": (-1, 1), "mild-": (-2, 0), "strong-": (-2, -1)}
viol = {}
for pid, p in P.items():
    if pid not in r: continue
    planted = {F[q.split(":")[1]] for q in p["planted"]}
    if p.get("background"): pass
    for t, z in p["hidden_slots"].items():
        if t in planted: continue                       # planted slots are meant to disagree
        lo, hi = OK[band(z)]; v = r[pid][t]
        if p.get("style") == "messy":            # short first turns only need the top type; nothing may point the wrong way
            lo, hi = (-2, 2) if abs(z) < 0.8 else ((0, 2) if z > 0 else (-2, 0))
        if not lo <= v <= hi: viol.setdefault(pid, []).append(f"{t} z={z:+.2f} ({band(z)}) rated {v:+g}")
    # the strongest type in the numbers must also be (joint) strongest in the text
    top = max(p["hidden_slots"], key=p["hidden_slots"].get)
    if top not in planted and r[pid][top] < max(r[pid].values()):
        viol.setdefault(pid, []).append(f"top type {top} not rated highest")
n = sum(len(v) for v in viol.values())
checked = sum(6 for p in P if p in r)
print(f"profiles failing: {len(viol)} / {len(P)}   slot violations: {n} / {checked}")
for k in sorted(viol): print(k, "|", "; ".join(viol[k]))
json.dump(sorted(viol), open("work/failing_ids.json", "w"))
