"""Review the negation probe set with single key presses (docs/LEXICAL_MATCHING_PLAN.md).

From the repo root:
    python -m devtools.review_probes                 # review, resuming where you stopped
    python -m devtools.review_probes --summary       # show what has been reviewed so far

For each probe you answer three yes/no questions:
  1. Is the wording something a student would plausibly say?
  2. Are the AVOIDED jobs right (the dislike should push these out of the top 10)?
  3. Are the ACCEPTABLE jobs right (the rest of the text points here; they should stay)?
Answering "no" to 2 or 3 opens a list where you untick wrong jobs by number. A note is optional.

Answers are saved after every probe to data/eval/negation_probes_v1_review.jsonl, one row per
probe and reviewer (a newer answer replaces an older one). The probe set itself is not changed:
apply agreed corrections in data/scripts/build_negation_probes_v1.py and rebuild.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
PROBES = ROOT / "data/eval/negation_probes_v1.jsonl"
REVIEW = ROOT / "data/eval/negation_probes_v1_review.jsonl"

os.system("")  # enables ANSI colours in the Windows console
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
B, DIM, G, R, Y, C, X = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"


def key(prompt: str, allowed: str) -> str:
    """One key press (no Enter) in a real terminal; one line when input is piped (for tests)."""
    print(prompt, end=" ", flush=True)
    while True:
        if sys.stdin.isatty() and os.name == "nt":
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow keys: left = back, right = skip
                ch = {"K": "b", "M": "s"}.get(msvcrt.getwch(), "")
        elif sys.stdin.isatty():
            import termios, tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        else:
            ch = (sys.stdin.readline() or "q").strip()[:1]
        ch = ch.lower()
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in allowed:
            print(ch)
            return ch


def line(prompt: str) -> str:
    print(prompt, end=" ", flush=True)
    return (sys.stdin.readline() if not sys.stdin.isatty() else input()).strip()


def load_reviews() -> dict:
    out = {}
    if REVIEW.exists():
        for row in map(json.loads, filter(str.strip, REVIEW.read_text(encoding="utf-8").splitlines())):
            out[(row["probe_id"], row["reviewer"])] = row
    return out


def save_reviews(reviews: dict) -> None:
    tmp = REVIEW.with_suffix(".tmp")
    rows = sorted(reviews.values(), key=lambda r: (r["probe_id"], r["reviewer"]))
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(REVIEW)  # atomic: a crash never leaves a half-written review file


def titles() -> dict:
    sys.path.insert(0, str(ROOT))
    from recommender.data.data_loader import load_eligible_occupations
    return {o.code: o.title for o in load_eligible_occupations(ROOT / "data/occupations.csv")}


def columns(items: list[str], width: int = 118) -> None:
    cell = min(max(len(s) for s in items) + 2, width // 2) if items else 0
    per_row = max(1, width // cell) if cell else 1
    for i in range(0, len(items), per_row):
        print("  " + "".join(s[:cell - 1].ljust(cell) for s in items[i:i + per_row]))


def untick(label: str, codes: list[str], names: dict) -> list[str]:
    """Show a numbered list; return the codes the reviewer marks as wrong."""
    print(f"\n{B}{label}{X} {DIM}(type the numbers of WRONG jobs, e.g. 3,7,12; Enter for none){X}")
    columns([f"{i:>2}. {names[c]}" for i, c in enumerate(codes, 1)])
    picked = line(">")
    wrong = []
    for part in picked.replace(" ", ",").split(","):
        if part.isdigit() and 1 <= int(part) <= len(codes):
            wrong.append(codes[int(part) - 1])
    if wrong:
        print(f"  {R}marked wrong:{X} " + "; ".join(names[c] for c in wrong))
    return wrong


def show(probe: dict, names: dict, index: int, total: int, previous: dict | None) -> None:
    print("\033[2J\033[H", end="")  # clear screen
    print(f"{B}{probe['id']}{X}  {DIM}{probe['split']} | {probe['kind']} | {index + 1}/{total}{X}"
          + (f"   {Y}already reviewed: {previous['verdict']}{X}" if previous else ""))
    if probe.get("note"):
        print(f"{DIM}note: {probe['note']}{X}")
    print(f"\n{B}User says{X}")
    for i, turn in enumerate(probe["turns"], 1):
        print(f"  {C}{i}. {turn}{X}")
    print(f"\n{B}Avoided{X} {DIM}(should leave the top 10) - {len(probe['avoided'])} jobs from rules: "
          f"{', '.join(r.split('|')[-1] for r in probe['avoid_rules'][:8])}{'...' if len(probe['avoid_rules']) > 8 else ''}{X}")
    if probe["avoided"]:
        columns([names[c] for c in probe["avoided"]])
    else:
        print(f"  {DIM}none (this probe checks that nothing is demoted){X}")
    print(f"\n{B}Acceptable{X} {DIM}(should stay){X}")
    if probe["acceptable"]:
        columns([names[c] for c in probe["acceptable"]])
    else:
        print(f"  {DIM}none (only the avoided list is scored){X}")
    print()


def review(probe: dict, names: dict, reviewer: str) -> dict | str:
    """Returns the review row, or 'back' / 'skip' / 'quit'."""
    nav = f"{DIM}[y] yes  [n] no  [b] back  [s] skip  [q] save & quit{X}"
    print(nav)
    answers, wrong_avoided, wrong_acceptable = {}, [], []
    for field, question in (("wording_ok", "1. Wording plausible for a student?"),
                            ("avoided_ok", "2. Avoided jobs right?"),
                            ("acceptable_ok", "3. Acceptable jobs right?")):
        if field == "avoided_ok" and not probe["avoided"]:
            answers[field] = True
            continue
        if field == "acceptable_ok" and not probe["acceptable"]:
            answers[field] = True
            continue
        ch = key(f"{B}{question}{X}", "ynbsq")
        if ch in "bsq":
            return {"b": "back", "s": "skip", "q": "quit"}[ch]
        answers[field] = ch == "y"
        if field == "avoided_ok" and ch == "n":
            wrong_avoided = untick("Avoided jobs", probe["avoided"], names)
        if field == "acceptable_ok" and ch == "n":
            wrong_acceptable = untick("Acceptable jobs", probe["acceptable"], names)
    note = ""
    if not all(answers.values()) or key(f"{B}Add a note?{X} {DIM}[y/n]{X}", "yn") == "y":
        note = line(f"{B}Note{X} {DIM}(e.g. better wording, a job to add; Enter to skip){X}:")
    verdict = "approved" if all(answers.values()) else "needs_change"
    return dict(probe_id=probe["id"], reviewer=reviewer, verdict=verdict, **answers,
                wrong_avoided=wrong_avoided, wrong_acceptable=wrong_acceptable, note=note,
                reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def summary(probes: list[dict], reviews: dict, names: dict) -> None:
    by_probe = {}
    for (pid, who), row in reviews.items():
        by_probe.setdefault(pid, []).append(row)
    done = sum(pid in by_probe for pid in (p["id"] for p in probes))
    print(f"{B}Reviewed {done}/{len(probes)} probes{X}  ({REVIEW})\n")
    for p in probes:
        rows = by_probe.get(p["id"], [])
        mark = (f"{G}approved{X}" if rows and all(r["verdict"] == "approved" for r in rows)
                else f"{Y}needs change{X}" if rows else f"{DIM}not reviewed{X}")
        print(f"  {p['id']} {p['split']:<11} {p['kind']:<14} {mark}  {DIM}{', '.join(r['reviewer'] for r in rows)}{X}")
        for r in rows:
            if r["verdict"] != "approved" or r["note"]:
                flags = [k[:-3] for k in ("wording_ok", "avoided_ok", "acceptable_ok") if not r[k]]
                print(f"      {r['reviewer']}: {', '.join(flags) or 'note'}"
                      + (f" | wrong avoided: {'; '.join(names[c] for c in r['wrong_avoided'])}" if r["wrong_avoided"] else "")
                      + (f" | wrong acceptable: {'; '.join(names[c] for c in r['wrong_acceptable'])}" if r["wrong_acceptable"] else "")
                      + (f" | note: {r['note']}" if r["note"] else ""))


def main():
    global REVIEW
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reviewer", default=None, help="Your name or initials (asked if omitted)")
    parser.add_argument("--summary", action="store_true", help="Show review progress and exit")
    parser.add_argument("--all", action="store_true", help="Start at the first probe, not the first unreviewed one")
    parser.add_argument("--review-file", type=Path, default=None, help=f"Where answers are saved (default {REVIEW.relative_to(ROOT)})")
    args = parser.parse_args()
    if args.review_file:
        REVIEW = args.review_file.resolve()
    probes = [json.loads(l) for l in PROBES.read_text(encoding="utf-8").splitlines() if l.strip()]
    names, reviews = titles(), load_reviews()
    if args.summary:
        summary(probes, reviews, names)
        return
    reviewer = args.reviewer or line(f"{B}Your name or initials{X} [{getpass.getuser()}]:") or getpass.getuser()
    i = 0 if args.all else next((k for k, p in enumerate(probes) if (p["id"], reviewer) not in reviews), len(probes))
    try:
        while 0 <= i < len(probes):
            probe = probes[i]
            show(probe, names, i, len(probes), reviews.get((probe["id"], reviewer)))
            result = review(probe, names, reviewer)
            if result == "quit":
                break
            if result == "back":
                i = max(0, i - 1)
                continue
            if result != "skip":
                reviews[(probe["id"], reviewer)] = result
                save_reviews(reviews)
            i += 1
    except KeyboardInterrupt:
        print()
    print("\033[2J\033[H", end="")
    summary(probes, reviews, names)


if __name__ == "__main__":
    main()
