"""Read transcript logs (logs/<participant code>.jsonl) as a readable, turn-by-turn report.

    .venv/bin/python -m devtools.view_logs                       # list every log and its sessions
    .venv/bin/python -m devtools.view_logs logs/TEST.jsonl       # every session in that log
    .venv/bin/python -m devtools.view_logs logs/TEST.jsonl --last  # only the latest session
    .venv/bin/python -m devtools.view_logs logs/TEST.jsonl --brief # one line per turn

Each turn shows what the user said, how the answer was read (alone, with the question, or by a rule),
what the extractor found, any contradiction checks with their NLI scores, the policy's decision and reason,
who wrote the reply (T5 or the question bank), the profile, the top jobs and the time each step took.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sessions(path: Path) -> list[list[dict]]:
    """Rows grouped by session, in file order."""
    groups: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row.get("session_id", "?"), []).append(row)
    return list(groups.values())


def _stance(r: dict) -> str:
    return (f"{r['slot']}{'+' if r['sign'] > 0 else '-'}{r['strength']:.2f} "
            f"(conf {r['confidence']:.2f}, {r['reason']} {r.get('reason_confidence', 0):.2f})")


def _profile(profile: dict | None) -> str:
    parts = [f"{t} {v['value']:+.2f}" for t, v in (profile or {}).items() if v.get("value") is not None]
    return ", ".join(parts) or "empty"


def show(session: list[dict], brief: bool = False) -> None:
    first = session[0]
    print(f"\n=== session {first.get('session_id', '?')[:8]}  started {first.get('ts', '')[:19]}  "
          f"mode {first.get('input_mode')}  components {first.get('components')}")
    for row in session:
        trace = row.get("trace") or {}
        if row["turn"] == 0:
            print(f"\n[T0] BOT: {row['reply']}")
            continue
        policy = trace.get("policy", {})
        if brief:
            print(f"[T{row['turn']}] {row['action']:9} {row['user_text'][:45]!r:48} -> {_profile(row.get('profile'))}")
            continue
        print(f"\n[T{row['turn']}] USER: {row['user_text']}")
        if trace:
            print(f"     read: {trace.get('answer_path', '?')}"
                  + (f"  (asked {trace.get('question_kind')} on {trace.get('asked_slot')})" if trace.get("asked_slot") else ""))
            print(f"     extractor: {'; '.join(map(_stance, trace.get('readings', []))) or 'no stance'}")
            if trace.get("question_attached"):
                print(f"     with question, raw: {'; '.join(map(_stance, trace.get('paired_raw', []))) or 'no stance'}")
            for check in trace.get("contradiction_checks", []):
                print(f"     contradiction {check['slot']}: NLI {check['nli_score']:.2f} -> "
                      f"{'KEPT' if check['kept'] else 'rejected'} | \"{check['earlier']}\" vs \"{check['now']}\"")
            if trace.get("held_contradictions"):
                print(f"     held contradictions: {', '.join(trace['held_contradictions'])}")
        print(f"     profile: {_profile(row.get('profile'))}")
        print(f"     policy: {row['action']}" + (f" {policy.get('slot')}" if policy.get("slot") else "")
              + (f" — {policy.get('why')}" if policy.get("why") else ""))
        print(f"     BOT ({trace.get('reply_source', '?')}): {row['reply']}")
        if trace.get("top5"):
            print(f"     top 5: {', '.join(t['title'] for t in trace['top5'])}")
        if trace.get("timings_ms"):
            print(f"     time: {', '.join(f'{k[:-3]} {v:.0f}ms' for k, v in trace['timings_ms'].items())}")
        for rec in row.get("recommendations") or []:
            print(f"       card {rec['rank']}. {rec['title']}  ({len(rec.get('tasks', []))} task quotes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", type=Path, help="a logs/<code>.jsonl file; omit to list all logs")
    parser.add_argument("--last", action="store_true", help="only the most recent session")
    parser.add_argument("--brief", action="store_true", help="one line per turn")
    args = parser.parse_args()
    if args.log is None:
        for path in sorted((ROOT / "logs").glob("*.jsonl")):
            groups = sessions(path)
            print(f"{path.relative_to(ROOT)}: {len(groups)} session(s), "
                  f"latest {groups[-1][0].get('ts', '')[:19] if groups else '-'}")
        return
    groups = sessions(args.log)
    for session in groups[-1:] if args.last else groups:
        show(session, args.brief)


if __name__ == "__main__":
    main()
