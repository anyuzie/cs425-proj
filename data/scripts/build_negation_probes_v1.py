"""Build data/eval/negation_probes_v1.jsonl: dislike probes for D1/D2 (docs/LEXICAL_MATCHING_PLAN.md).

Drafted 9 November 2026; reviewed by the owner (aanny) on 9 November 2026 with
devtools/review_probes.py (answers in data/eval/negation_probes_v1_review.jsonl): 23 probes approved
as drafted; N07 corrected (device/technical roles are not "working with patients"). Each probe is one or two user turns with a
dislike, the occupations the dislike should push out of the top 10 (``avoided``) and occupations
the rest of the text points to, which must stay (``acceptable``). Avoided families are title rules
so they are easy to review; the builder resolves them against the 378 eligible occupations.

Split: ``calibration`` may be used to set D2's threshold; ``test`` is for the final claim only.
Probe text and lists were written from occupation titles only, without looking at any model
ranking. EVALUATION ONLY outside the calibration split: never tune on ``test``.

Kinds:
- explicit / casual / extractor_miss: the dislike names a job family; casual uses apostrophe-less
  forms, extractor_miss uses wording the explicit_scoped_v1 extractor does not label negative.
- narrow: the dislike removes one specialty inside the field the user likes.
- collateral: the negative clause names no job; nothing should be demoted (avoided is empty).
- only_negative: no positive text; only the avoided metric applies.
- control: no dislike; regression check for positive matching.

Usage: python data/scripts/build_negation_probes_v1.py [--check]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recommender.data.data_loader import load_eligible_occupations  # noqa: E402

OUTPUT = ROOT / "data/eval/negation_probes_v1.jsonl"

# Job families, as case-insensitive title substrings. "29-|Nurse" also requires the code to start with
# "29-" (SOC 29, healthcare practitioners), so counselling jobs such as Marriage and Family Therapists
# (SOC 21) are not swept into a healthcare family.
TEACHING = ["Teacher", "Instructors", "Teaching Assistants", "Educators", "Adapted Physical Education Specialists"]
SALES = ["Sales", "Real Estate Brokers"]
LAB = ["Laboratory", "Cytogenetic Technologists", "Cytotechnologists", "Histotechnologists", "Biological Technicians"]
NURSING = ["Nurse"]
SURGERY = ["Surgeon"]
PATIENT_FACING = ["29-|" + s for s in ["Nurse", "Physician", "Surgeon", "Therapist", "Dentist", "Orthodontist", "Prosthodontist",
                  "Chiropractor", "Optometrist", "Podiatrist", "Pediatrician", "Psychiatrist", "Cardiologist",
                  "Dermatologist", "Neurologist", "Obstetrician", "Radiologist", "Urologist", "Allergist",
                  "Hospitalist", "Anesthesiologist", "Audiologist", "Midwi", "Athletic Trainer", "Acupuncturist",
                  "Speech-Language", "Ophthalmologist", "Orthoptist", "Dietitian", "Exercise Physiologist",
                  "Genetic Counselor", "Orthotist"]]
CHILDREN_TEACHING = ["Kindergarten Teachers", "Elementary School Teachers", "Middle School Teachers",
                     "Special Education Teachers", "Adapted Physical Education Specialists"]
SECURITY_IT = ["Information Security", "Penetration Testers", "Digital Forensics", "Security Management Specialists"]
MUSIC = ["Music Directors and Composers", "Musicians and Singers", "Music Therapists"]  # not "Art, Drama, and Music Teachers"
MANAGING_IT = ["Computer and Information Systems Managers", "Information Technology Project Managers"]

PROBES = [
    # ---- calibration ----
    dict(id="N01", split="calibration", kind="explicit", turns=["I study psychology at NUS", "i like helping others. but teaching, not as much"],
         avoid=TEACHING, acceptable=["Clinical and Counseling Psychologists", "Mental Health Counselors", "Marriage and Family Therapists"],
         note="owner's browser case (anon-c160cd58)"),
    dict(id="N02", split="calibration", kind="narrow", turns=["I love biology, just not lab work."],
         avoid=LAB, acceptable=["Zoologists and Wildlife Biologists", "Conservation Scientists", "Park Naturalists"]),
    dict(id="N03", split="calibration", kind="explicit", turns=["I enjoy working with numbers, but I don't want to do sales."],
         avoid=SALES, acceptable=["Accountants and Auditors", "Budget Analysts", "Statisticians", "Actuaries"]),
    dict(id="N04", split="calibration", kind="casual", turns=["i like computers and coding. dont want to manage people tho"],
         avoid=MANAGING_IT, acceptable=["Software Developers", "Computer Programmers", "Database Architects"]),
    dict(id="N05", split="calibration", kind="extractor_miss", turns=["Teaching is not for me. I'm really into history and old documents."],
         avoid=TEACHING, acceptable=["Historians", "Archivists", "Curators"]),
    dict(id="N06", split="calibration", kind="extractor_miss", turns=["Anything but healthcare. I like writing and telling stories."],
         avoid=PATIENT_FACING, acceptable=["Writers and Authors", "Poets, Lyricists and Creative Writers", "Editors"]),
    dict(id="N07", split="calibration", kind="extractor_miss", turns=["I'd rather not work with patients. I like chemistry and running experiments."],
         avoid=PATIENT_FACING, acceptable=["Chemists", "Materials Scientists", "Biochemists and Biophysicists"],
         avoid_except=["Orthotists and Prosthetists", "Anesthesiologist Assistants", "Low Vision Therapists"],
         note="review 2026-11-09: device/technical roles removed from avoided"),
    dict(id="N08", split="calibration", kind="collateral", turns=["I love composing music. I don't really care about how things look."],
         avoid=[], acceptable=["Music Directors and Composers", "Musicians and Singers"]),
    dict(id="N09", split="calibration", kind="narrow", turns=["I want to work in law, but I never want to be a judge."],
         avoid=["Judges"], acceptable=["Lawyers", "Arbitrators, Mediators, and Conciliators"]),
    dict(id="N10", split="calibration", kind="narrow", turns=["I'm into engineering, just not anything to do with cars."],
         avoid=["Automotive Engineers"], acceptable=["Mechanical Engineers", "Aerospace Engineers", "Robotics Engineers"]),
    dict(id="N11", split="calibration", kind="only_negative", turns=["I hate sales and I don't want to teach."],
         avoid=SALES + TEACHING, acceptable=[]),
    dict(id="N12", split="calibration", kind="control", turns=["I love designing posters and logos for small businesses."],
         avoid=[], acceptable=["Graphic Designers", "Art Directors"]),
    # ---- test ----
    dict(id="N13", split="test", kind="casual", turns=["i love kids but teaching isnt for me"],
         avoid=CHILDREN_TEACHING, acceptable=["Child, Family, and School Social Workers", "Pediatricians, General", "School Psychologists"]),
    dict(id="N14", split="test", kind="explicit", turns=["I enjoy finance, but I don't want to sell anything."],
         avoid=SALES, acceptable=["Financial and Investment Analysts", "Financial Risk Specialists", "Budget Analysts"]),
    dict(id="N15", split="test", kind="extractor_miss", turns=["Nursing is not for me. I want to help people with their mental health."],
         avoid=NURSING, acceptable=["Mental Health Counselors", "Clinical and Counseling Psychologists", "Psychiatrists"]),
    dict(id="N16", split="test", kind="explicit", turns=["I like the environment and being in nature. I would never want to work in a lab."],
         avoid=LAB, acceptable=["Conservation Scientists", "Park Naturalists", "Foresters"]),
    dict(id="N17", split="test", kind="casual", turns=["i like computers. cant stand cybersecurity stuff though"],
         avoid=SECURITY_IT, acceptable=["Software Developers", "Database Architects", "Computer Systems Analysts"]),
    dict(id="N18", split="test", kind="extractor_miss", turns=["Anything but accounting. I'm good with numbers and I like research."],
         avoid=["Accountants and Auditors"], acceptable=["Statisticians", "Mathematicians", "Economists"]),
    dict(id="N19", split="test", kind="narrow", turns=["I'm interested in medicine, but I don't want to do surgery."],
         avoid=SURGERY, acceptable=["Family Medicine Physicians", "General Internal Medicine Physicians", "Pediatricians, General"]),
    dict(id="N20", split="test", kind="narrow", turns=["I like art, just not music."],
         avoid=MUSIC, acceptable=["Graphic Designers", "Art Directors", "Special Effects Artists and Animators"]),
    dict(id="N21", split="test", kind="casual", turns=["i enjoy managing teams. i dont want anything to do with construction"],
         avoid=["Construction Managers"], acceptable=["General and Operations Managers", "Human Resources Managers"]),
    dict(id="N22", split="test", kind="collateral", turns=["I want to work with animals. I'm not a morning person."],
         avoid=[], acceptable=["Veterinarians", "Zoologists and Wildlife Biologists", "Animal Scientists"]),
    dict(id="N23", split="test", kind="only_negative", turns=["I don't want to work with patients or children."],
         avoid=PATIENT_FACING + CHILDREN_TEACHING, acceptable=[]),
    dict(id="N24", split="test", kind="control", turns=["I love analysing data and building statistical models."],
         avoid=[], acceptable=["Statisticians", "Data Scientists", "Biostatisticians"]),
]


def matches(rule: str, occupation) -> bool:
    prefix, _, text = rule.rpartition("|")
    return occupation.code.startswith(prefix) and text.lower() in occupation.title.lower()


def build():
    occupations = load_eligible_occupations(ROOT / "data/occupations.csv")
    by_title = {o.title: o.code for o in occupations}
    rows, problems = [], []
    for p in PROBES:
        avoided = sorted(o.code for o in occupations if any(matches(s, o) for s in p["avoid"])
                         and not any(e.lower() in o.title.lower() for e in p.get("avoid_except", [])))
        unused_except = [e for e in p.get("avoid_except", []) if not any(e.lower() in o.title.lower() for o in occupations)]
        if unused_except:
            problems.append(f"{p['id']}: avoid_except matches nothing: {unused_except}")
        missing = [t for t in p["acceptable"] if t not in by_title]
        acceptable = [by_title[t] for t in p["acceptable"] if t in by_title]
        unused = [s for s in p["avoid"] if not any(matches(s, o) for o in occupations)]
        if missing:
            problems.append(f"{p['id']}: acceptable titles not eligible: {missing}")
        if unused:
            problems.append(f"{p['id']}: avoid rules match nothing: {unused}")
        if set(avoided) & set(acceptable):
            problems.append(f"{p['id']}: acceptable occupations are also avoided")
        if p["kind"] in ("collateral", "control") and avoided:
            problems.append(f"{p['id']}: {p['kind']} probe must not avoid anything")
        rows.append(dict(id=p["id"], split=p["split"], kind=p["kind"], turns=p["turns"],
                         avoid_rules=p["avoid"], avoid_except=p.get("avoid_except", []), avoided=avoided, acceptable_titles=p["acceptable"], acceptable=acceptable,
                         note=p.get("note"), source="drafted 2026-11-09 from occupation titles only; owner-reviewed 2026-11-09"))
    return rows, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Build and report without writing")
    args = parser.parse_args()
    rows, problems = build()
    for r in rows:
        print(f"{r['id']} {r['split']:<11} {r['kind']:<14} avoided={len(r['avoided']):>3} acceptable={len(r['acceptable'])}  {r['turns'][-1][:60]}")
    print(f"{len(rows)} probes; " + ", ".join(f"{s}={sum(r['split'] == s for r in rows)}" for s in ("calibration", "test")))
    for p in problems:
        print("PROBLEM:", p)
    if args.check or problems:
        return
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    OUTPUT.write_text(text, encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT), "sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
