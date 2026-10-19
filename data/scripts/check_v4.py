"""Band check for v4 profiles: blind ratings (-2..+2) against the hidden z bands (same rules as check_bands.py)."""
import json, glob
W = "/home/claude/data/work/v4"
F = {"Realistic": "R", "Investigative": "I", "Artistic": "A", "Social": "S", "Enterprising": "E", "Conventional": "C"}
key = json.load(open(f"{W}/rate_key.json"))
r = {}
for f in sorted(glob.glob(f"{W}/rate_out_*.json")):
    for q, v in json.load(open(f)).items(): r[key[q]] = v
m1 = json.load(open(f"{W}/tier1_meta.json")); m2 = {x["id"]: x for x in json.load(open(f"{W}/tier2_meta.json"))}
d2 = {}
for f in sorted(glob.glob(f"{W}/t2_out_*.json")): d2.update({x["id"]: x for x in json.load(open(f))})
def band(z):
    return "strong+" if z >= 1.0 else "mild+" if z >= 0.3 else "neutral" if z > -0.4 else "mild-" if z > -0.8 else "strong-"
OK = {"strong+": (1, 2), "mild+": (0, 2), "neutral": (-1, 1), "mild-": (-2, 0), "strong-": (-2, -1)}
viol = {}
for pid in list(m1) + list(m2):
    if pid not in r: continue
    m = m1.get(pid) or m2[pid]; h = m["hidden_slots"]
    planted = {F[q.split(":")[1]] for q in d2[pid]["planted"]} if pid in d2 else set()
    for t, z in h.items():
        if t in planted: continue
        lo, hi = OK[band(z)]; v = r[pid][t]
        if m.get("style") == "messy":
            lo, hi = (-2, 2) if abs(z) < 0.8 else ((0, 2) if z > 0 else (-2, 0))
        if not lo <= v <= hi: viol.setdefault(pid, []).append(f"{t} z={z:+.2f} ({band(z)}) rated {v:+g}")
    top = max(h, key=h.get)
    if top not in planted and r[pid][top] < max(r[pid][t] for t in h if t not in planted):
        viol.setdefault(pid, []).append(f"top type {top} not rated highest")
n = sum(len(v) for v in viol.values())
print(f"rated {len(r)} | profiles failing: {len(viol)} | slot violations: {n} / {6 * len(r)}")
for k in sorted(viol): print(k, m1.get(k, m2.get(k, {})).get("pattern"), "|", "; ".join(viol[k]))
json.dump(r, open(f"{W}/ratings.json", "w"))
json.dump(viol, open(f"{W}/violations.json", "w"), indent=1)
