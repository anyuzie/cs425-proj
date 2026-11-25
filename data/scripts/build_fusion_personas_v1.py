"""Build data/eval/fusion_personas_v1.jsonl: scripted personas for the fusion evaluation.

Evaluation only: never tune any component, weight or threshold on these personas.

Written by hand: opening text, hidden RIASEC slots, per-slot resolutions, field, intended
conflict and the acceptable-occupation list.
Computed here: target (first acceptable code), target_profile, O*NET related lists, and the
observed conflict label.

Conflict label (computed, not authored): rank of the best acceptable occupation under oracle
cosine (hidden_slots, all six types observed, full eligible pool). conflict = rank > CONFLICT_RANK.
A persona whose intent and observed label disagree is reported, not silently relabelled.

Persona text was written without looking at any semantic ranking, so the set does not
favour the semantic component.

Usage: python data/scripts/build_fusion_personas_v1.py [--check]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from recommender.components import ComponentInput  # noqa: E402
from recommender.data.data_loader import load_eligible_occupations  # noqa: E402
from recommender.models.cosine import BASELINE_PARAMETERS, CosineSimilarity  # noqa: E402
from recommender.schemas import TYPES  # noqa: E402

OUTPUT = ROOT / "data/eval/fusion_personas_v1.jsonl"
CONFLICT_RANK = 20  # about the top 5% of the 378-occupation pool
UNSURE = "Not sure about that one, I haven't really thought about it."
LEVEL = lambda z: "pos" if z >= 1 else "pos_mild" if z >= 0.4 else "dontmind" if z > -0.4 else "neg_mild" if z > -0.8 else "neg"


def slots(R, I, A, S, E, C):
    return dict(R=R, I=I, A=A, S=S, E=E, C=C)


# fmt: off
PERSONAS = [
    # intended conflict: stated topic and RIASEC profile point different ways
    dict(id="F01", field="music", conflict_intended=True, style="full",
         text="I've been writing music since I was twelve, mostly arranging pieces for small string groups and lately scoring short student films. The part I love is sitting with a score and working out why a harmony works, I could analyse that for hours. I'm not a performer and crowds drain me, and I hate having to sell myself. I want a career where music is the centre of it.",
         hidden=slots(-0.9, 1.4, 1.6, -0.9, -1.2, 0.2),
         res=dict(R="I'm not really into building or fixing things.", I="I love analysing why a piece of music works.", A="Writing and arranging music is what I do most.", S="Working closely with lots of people all day drains me.", E="I really don't like selling myself or persuading people.", C="Keeping things organised is fine, I don't mind it."),
         acceptable=["27-2041.00", "27-2042.00", "25-1121.00", "29-1129.02"]),
    dict(id="F02", field="veterinary", conflict_intended=True, style="full",
         text="I want to work with animals, ideally as a vet. I volunteer at a rescue shelter every weekend and the best part is calming the scared dogs and talking worried owners through what happens next. I'm okay with hands-on stuff like cleaning kennels. Science classes were never my strong point and I don't love studying.",
         hidden=slots(0.5, -0.7, 0.2, 1.8, -0.2, -0.6),
         res=dict(R="Hands-on work like cleaning kennels is okay with me.", I="Science and studying were never really my thing.", A=UNSURE, S="Calming animals and reassuring their owners is what I love.", E=UNSURE, C="I'm not big on routine paperwork."),
         acceptable=["29-1131.00", "19-1011.00", "19-1023.00"]),
    dict(id="F03", field="history_heritage", conflict_intended=True, style="full",
         text="History is my favourite thing. I give tours at a local stately home on weekends and I love telling visitors the stories behind each room and getting them excited about the past. I'd like to plan events and exhibitions one day. Sitting alone cataloguing documents sounds awful to me though.",
         hidden=slots(-0.6, 0.3, 0.9, 1.6, 1.4, -1.0),
         res=dict(R="I'm not into manual or mechanical work.", I="I like reading about the past but I'm not a researcher.", A="Telling stories and designing exhibitions sounds fun.", S="Showing visitors around and getting them excited is the best part.", E="I'd like to organise events and lead projects one day.", C="Sitting alone cataloguing documents sounds awful."),
         acceptable=["19-3093.00", "25-4011.00", "25-4012.00", "25-4013.00", "25-1125.00"]),
    dict(id="F04", field="law", conflict_intended=True, style="full",
         text="I'm fascinated by law, especially reading court judgements and following how judges reason their way through a case. I joined a summer reading group on constitutional cases and loved writing up careful analyses. I'm quiet, and I hate arguing with people or trying to win someone over. Practical hands-on work isn't for me.",
         hidden=slots(-1.0, 1.6, 1.3, -0.5, -1.0, 0.4),
         res=dict(R="Practical hands-on work isn't for me.", I="I love working through how judges reasoned in a case.", A="Writing careful analyses is something I really enjoy.", S="I'm quiet and don't love being around people all day.", E="I hate arguing with people or trying to win them over.", C="Keeping detailed notes is fine with me."),
         acceptable=["23-1011.00", "23-1012.00", "23-1021.00", "23-1023.00", "25-1112.00"]),
    dict(id="F05", field="video_games", conflict_intended=True, style="messy",
         text="i want to make video games!! mostly i love the mechanics side, balancing systems, tuning numbers so a level feels fair, i keep spreadsheets of drop rates lol. cant draw at all tho",
         hidden=slots(-0.3, 1.7, -1.0, -0.6, -0.2, 1.3),
         res=dict(R=UNSURE, I="I like figuring out how game systems work under the hood.", A="I can't draw at all, I'm not artistic.", S="I'd rather work on my own than with big groups.", E=UNSURE, C="I like keeping spreadsheets and tracking numbers."),
         acceptable=["15-1255.01", "27-1014.00"]),
    dict(id="F06", field="climate_policy", conflict_intended=True, style="full",
         text="Climate change is the issue I care about most. I organised my school's climate strike and spent last summer lobbying our council for safer bike lanes. I love getting people together around a cause and persuading the people in charge. I'm honestly not a lab or numbers person.",
         hidden=slots(-0.5, -0.6, 0.3, 1.3, 1.7, -0.2),
         res=dict(R="I'm not really into hands-on technical work.", I="I'm not a lab or numbers person.", A=UNSURE, S="Bringing people together around a cause is what I love.", E="Persuading the people in charge is where I'm at my best.", C=UNSURE),
         acceptable=["19-2041.01", "13-1199.05", "11-1011.03", "19-2041.02"]),
    dict(id="F07", field="surgery", conflict_intended=True, style="full",
         text="I want to be a surgeon. I love precise work with my hands: I build model engines and I'm good at fine soldering. I like a clear procedure and doing it exactly right. Bedside chat and comforting people isn't really my thing, I'd rather just fix the problem.",
         hidden=slots(1.6, 0.5, -0.3, -0.9, -0.2, 0.6),
         res=dict(R="Precise work with my hands is what I love.", I="I'm interested in how the body works.", A=UNSURE, S="Comforting people and chatting isn't really my thing.", E=UNSURE, C="I like following a clear procedure exactly."),
         acceptable=["29-1242.00", "29-1243.00", "29-1022.00"]),
    dict(id="F08", field="food", conflict_intended=True, style="messy",
         text="food is my thing honestly. i bake constantly, make up my own recipes and run a little food account where i post everything. love hosting dinners for friends too. want a career around food somehow",
         hidden=slots(0.4, -0.5, 1.6, 0.8, 1.0, -0.6),
         res=dict(R="I like cooking and making things with my hands.", I="I'm not really into science or research.", A="Inventing recipes and posting them is my favourite thing.", S="I love hosting dinners for friends.", E="Growing my food account has been really fun.", C="I'm not big on routine admin."),
         acceptable=["19-1012.00", "29-1031.00", "25-1192.00"]),
    dict(id="F09", field="journalism", conflict_intended=True, style="full",
         text="I want to work in news. I read three papers a day, fact-check viral claims for fun and love digging through public records to find out what actually happened. I'm not a creative writer and I definitely don't want to be on camera or pitching stories to people.",
         hidden=slots(-0.5, 1.5, -0.9, 0.2, -0.9, 1.0),
         res=dict(R="I'm not into hands-on or outdoor work.", I="Digging into records to find out what happened is what I love.", A="I'm not a creative writer.", S=UNSURE, E="I don't want to be on camera or pitching to people.", C="I like checking facts carefully and keeping records straight."),
         acceptable=["27-3023.00", "27-3041.00"]),
    dict(id="F10", field="architecture", conflict_intended=True, style="full",
         text="I want to be an architect. Buildings fascinate me, especially how the structure actually holds up. I spend weekends on building sites with my uncle and I like reading through technical drawings and checking measurements. I'm not artistic at all and sketching isn't something I enjoy.",
         hidden=slots(1.2, 0.4, -0.6, -0.3, 0.2, 1.4),
         res=dict(R="Being on building sites and hands-on work suits me.", I="I like understanding how structures hold up.", A="I'm not artistic and don't enjoy sketching.", S=UNSURE, E=UNSURE, C="Checking drawings and measurements carefully is satisfying."),
         acceptable=["17-1011.00", "17-1012.00", "25-1031.00"]),
    dict(id="F11", field="aviation", conflict_intended=True, style="full",
         text="Ever since I was small I've wanted to fly planes. I've done a few hours in a glider and I read about aerodynamics and weather systems constantly. I'm not someone who wants to be in charge of people or make speeches.",
         hidden=slots(0.3, 1.8, 0.6, -0.5, -1.0, -0.2),
         res=dict(R=UNSURE, I="I read about aerodynamics and weather all the time.", A="I like photography from the air, it's kind of creative.", S="I'm not a big people person.", E="I don't want to be in charge of people or make speeches.", C=UNSURE),
         acceptable=["53-2011.00"]),
    dict(id="F12", field="religion", conflict_intended=True, style="full",
         text="Faith is what I want to spend my life on. I read theology and the history of religious texts constantly and I'm seriously thinking about ministry. I'm quite introverted and I find big social events and persuading people exhausting.",
         hidden=slots(-0.5, 1.8, 1.0, -0.6, -0.8, 0.5),
         res=dict(R="Manual work isn't really for me.", I="I read theology and the history of religious texts all the time.", A="I enjoy writing reflections on what I read.", S="I'm introverted and big social events exhaust me.", E="Persuading people is exhausting for me.", C="I like keeping my notes well organised."),
         acceptable=["21-2011.00", "21-2021.00", "25-1126.00"]),
    dict(id="F13", field="finance", conflict_intended=True, style="full",
         text="I want to work in finance. I follow the stock market every day and run a mock portfolio with my friends. I love talking ideas through with people and coming up with creative angles on a company. Checking numbers line by line and repetitive paperwork bore me to tears.",
         hidden=slots(0.2, 0.3, 1.4, 1.0, 0.6, -1.3),
         res=dict(R=UNSURE, I=UNSURE, A="Coming up with creative angles on ideas is fun.", S="I love talking ideas through with people.", E="Running the mock portfolio with friends is exciting.", C="Checking numbers line by line bores me to tears."),
         acceptable=["13-2051.00", "11-3031.03", "41-3031.00", "13-2052.00"]),
    dict(id="F14", field="sport", conflict_intended=True, style="messy",
         text="football is my life. i track every stat for my team, build spreadsheets on player performance and watch match footage frame by frame. not a people person and i dont want to be shouting from the sidelines",
         hidden=slots(0.3, 1.6, -0.4, -0.8, -0.3, 1.2),
         res=dict(R=UNSURE, I="Breaking down match footage and stats is what I love.", A="I'm not really a creative person.", S="I'm not a people person.", E=UNSURE, C="Building spreadsheets of player stats is fun for me."),
         acceptable=["27-2022.00"]),

    # intended agree: stated topic and RIASEC profile point the same way
    dict(id="F15", field="nursing", conflict_intended=False, style="full",
         text="I want to be a nurse. I looked after my grandmother through her illness and found I liked learning the medical side, why each medicine was given and what to watch for. I stay calm in a crisis and I'm good at keeping track of details. Art and performing aren't me.",
         hidden=slots(-0.4, 0.9, -0.5, 1.7, -0.2, 0.3),
         res=dict(R="I'm not into mechanical work.", I="Learning why each medicine was given was really interesting.", A="Art and performing aren't me.", S="Caring for my grandmother showed me I want to look after people.", E=UNSURE, C=UNSURE),
         acceptable=["29-1141.00", "29-1171.00", "29-1141.01", "29-1141.03", "29-1141.04"]),
    dict(id="F16", field="software", conflict_intended=False, style="full",
         text="I build small apps in my free time, like a bus timetable app and a bot for my friends' group chat. I love debugging and figuring out exactly why something breaks, and I like keeping my code tidy and organised. I'd rather avoid sales or lots of meetings.",
         hidden=slots(-0.3, 1.7, 0.3, -0.7, -0.6, 1.0),
         res=dict(R=UNSURE, I="Figuring out exactly why something breaks is my favourite part.", A=UNSURE, S="I'd rather avoid lots of meetings.", E="I'd rather stay away from sales.", C="I like keeping my code tidy and organised."),
         acceptable=["15-1252.00", "15-1251.00", "15-1253.00"]),
    dict(id="F17", field="counselling", conflict_intended=False, style="full",
         text="Friends always come to me with their problems and I love helping them work through things. I read a lot about psychology and different therapy approaches, and I journal every night. Fixing machines or doing admin isn't for me.",
         hidden=slots(-1.5, 0.8, 0.7, 2.0, 0.2, -1.0),
         res=dict(R="Fixing machines isn't for me.", I="I read a lot about psychology and therapy approaches.", A="Journaling and writing are things I enjoy.", S="Helping friends work through their problems is what I love.", E=UNSURE, C="Admin and paperwork aren't for me."),
         acceptable=["21-1014.00", "19-3033.00", "21-1013.00", "21-1011.00", "21-1023.00"]),
    dict(id="F18", field="accounting", conflict_intended=False, style="full",
         text="I like numbers being exactly right. I do the books for my parents' shop and actually enjoy reconciling everything at the end of the month. Eventually I'd like to move into the business side and help run things. I'm not creative and I don't want hands-on physical work.",
         hidden=slots(-1.4, 0.3, -0.8, -0.3, 0.7, 2.2),
         res=dict(R="I don't want hands-on physical work.", I=UNSURE, A="I'm not creative.", S=UNSURE, E="I'd like to help run a business one day.", C="Reconciling the books so everything is exactly right is satisfying."),
         acceptable=["13-2011.00", "13-2031.00", "13-2061.00", "11-3031.01"]),
    dict(id="F19", field="marketing", conflict_intended=False, style="full",
         text="I ran the social media for my uni society and grew it a lot. I love working out what makes people click, pitching campaign ideas and leading the team that pulls them off. Some design work is fun too. Lab work and fixing things don't interest me.",
         hidden=slots(-1.6, -0.6, 0.7, 0.0, 2.3, 0.2),
         res=dict(R="Fixing things doesn't interest me.", I="Lab work doesn't interest me much.", A="Some design work is fun.", S=UNSURE, E="Pitching campaign ideas and leading the team is what I love.", C=UNSURE),
         acceptable=["11-2021.00", "11-2011.00", "13-1161.00", "13-1161.01", "27-3031.00"]),
    dict(id="F20", field="primary_teaching", conflict_intended=False, style="full",
         text="I want to teach primary school. I help at a Saturday kids' club doing arts and crafts and reading with the children, and seeing them get something for the first time is the best feeling. I'm not into machines or office routine.",
         hidden=slots(-0.8, 0.1, 1.5, 2.2, -0.3, -0.6),
         res=dict(R="I'm not into machines.", I=UNSURE, A="Arts and crafts with the kids is so much fun.", S="Seeing children get something for the first time is the best feeling.", E=UNSURE, C="Office routine isn't for me."),
         acceptable=["25-2021.00", "25-2012.00", "25-2056.00", "25-2055.00", "25-2022.00"]),
    dict(id="F21", field="geology", conflict_intended=False, style="full",
         text="I collect rocks and fossils and I love fieldwork, hiking out to map outcrops and working out how a landscape formed. I'd happily spend weeks outdoors taking samples. Selling things or managing people doesn't appeal to me.",
         hidden=slots(0.7, 2.0, 0.4, -0.5, -0.6, -0.2),
         res=dict(R="Hiking out and taking samples outdoors is great.", I="Working out how a landscape formed is fascinating.", A="Sketching outcrops in my field notebook is fun.", S="I'm not big on working with crowds of people.", E="Selling things or managing people doesn't appeal to me.", C=UNSURE),
         acceptable=["19-2042.00", "19-2043.00", "19-4043.00", "17-2151.00", "25-1051.00"]),
    dict(id="F22", field="graphic_design", conflict_intended=False, style="messy",
         text="i do posters and logos for local bands and small shops, been designing since school. love playing with type and colour. dont care much for science stuff",
         hidden=slots(-0.5, -0.2, 2.4, -0.3, 0.4, -0.5),
         res=dict(R="I'm not into mechanical stuff.", I=UNSURE, A="Playing with type and colour is what I love.", S=UNSURE, E="Pitching designs to shop owners is kind of fun.", C="Routine admin isn't for me."),
         acceptable=["27-1024.00", "27-1011.00", "15-1255.00", "27-1014.00"]),
    dict(id="F23", field="robotics", conflict_intended=False, style="full",
         text="I'm on my school's robotics team and I love building the robot, wiring the motors and working out why the arm keeps overshooting. I like solving technical problems more than anything. Social events and selling aren't really my scene.",
         hidden=slots(1.0, 1.3, 0.1, -0.8, -0.5, 0.1),
         res=dict(R="Building the robot and wiring motors is what I love.", I="Working out why the arm overshoots is the fun part.", A=UNSURE, S="Social events aren't really my scene.", E="Selling isn't my scene.", C=UNSURE),
         acceptable=["17-2199.08", "17-2141.00", "17-2199.05", "17-2141.02"]),
    dict(id="F24", field="statistics", conflict_intended=False, style="full",
         text="Data is my favourite thing. I love taking a messy dataset, cleaning it up and finding out what's actually going on underneath. I did a project modelling bus delays in my city and loved it. I'm not into hands-on work or anything involving lots of persuading.",
         hidden=slots(-1.0, 1.9, 0.1, -0.6, -0.6, 1.3),
         res=dict(R="I'm not into hands-on work.", I="Finding out what's going on underneath the data is what I love.", A=UNSURE, S="I prefer working on my own.", E="I'm not into anything involving lots of persuading.", C="Cleaning up a messy dataset is weirdly satisfying."),
         acceptable=["15-2041.00", "15-2051.00", "15-2041.01", "15-2031.00", "15-2011.00"]),
    dict(id="F25", field="sales_property", conflict_intended=False, style="messy",
         text="i love closing a deal lol. sold stuff online all through school and now i help my aunt show houses. good with paperwork and contracts too. science and art not really me",
         hidden=slots(-0.8, -0.8, -0.5, 0.2, 2.0, 1.0),
         res=dict(R="Hands-on manual work isn't for me.", I="Science isn't really me.", A="Art isn't really me.", S=UNSURE, E="Closing a deal is the best feeling.", C="I'm good with paperwork and contracts."),
         acceptable=["41-9021.00", "11-2022.00", "41-3021.00", "11-9141.00", "41-3031.00"]),
]
# fmt: on


def related_lists():
    out = defaultdict(lambda: dict(primary_short=[], primary_long=[], supplemental=[]))
    key = {"Primary-Short": "primary_short", "Primary-Long": "primary_long", "Supplemental": "supplemental"}
    with open(ROOT / "data/related_occupations.csv", encoding="utf-8") as f:
        for row in sorted(csv.DictReader(f), key=lambda r: int(r["rank"])):
            out[row["code"]][key[row["tier"]]].append(row["related_code"])
    return out


def build():
    occupations = load_eligible_occupations(ROOT / "data/occupations.csv")
    pool = {o.code: o for o in occupations}
    cosine = CosineSimilarity(occupations, BASELINE_PARAMETERS)
    related = related_lists()
    if len({p["id"] for p in PERSONAS}) != len(PERSONAS):
        raise ValueError("duplicate persona id")
    rows, problems = [], []
    for p in PERSONAS:
        missing = [c for c in p["acceptable"] if c not in pool]
        if missing:
            raise ValueError(f"{p['id']}: acceptable codes outside eligible pool: {missing}")
        for t in TYPES:  # an 'unsure' resolution is only consistent with a 'dontmind' slot
            if p["res"][t] == UNSURE and LEVEL(p["hidden"][t]) != "dontmind":
                problems.append(f"{p['id']}: slot {t} level {LEVEL(p['hidden'][t])} vs resolution {p['res'][t]!r}")
        ranking = cosine.score(ComponentInput(dict(p["hidden"]), frozenset(TYPES), ()))
        rank = {s.code: s.rank for s in ranking.occupations}
        best = min(rank[c] for c in p["acceptable"])
        target = p["acceptable"][0]
        conflict = best > CONFLICT_RANK
        if conflict != p["conflict_intended"]:
            problems.append(f"{p['id']}: intended conflict={p['conflict_intended']} but oracle best rank={best}")
        rel = related[target]
        rows.append(dict(
            id=p["id"], field=p["field"], style=p["style"], text=p["text"],
            planted=[], contradiction=None, latent_reveal="",
            hidden_slots=p["hidden"], resolutions=p["res"],
            target=target, target_profile={t: round(pool[target].z_scores[t], 2) for t in TYPES},
            acceptable=p["acceptable"],
            conflict=conflict, conflict_intended=p["conflict_intended"],
            oracle_cosine_best_rank=best,
            oracle_cosine_ranks={c: rank[c] for c in p["acceptable"]},
            related=rel,
            related_in_pool={k: [c for c in v if c in pool] for k, v in rel.items()},
            source="hand-drafted and team-reviewed 2026-11-09; evaluation only, never tune",
        ))
    return rows, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Build and report without writing")
    args = parser.parse_args()
    rows, problems = build()
    for r in rows:
        print(f"{r['id']} {r['field']:<17} conflict={str(r['conflict']):<5} oracle_best={r['oracle_cosine_best_rank']:>3}  ranks={r['oracle_cosine_ranks']}")
    print(f"{sum(r['conflict'] for r in rows)} conflict / {len(rows)} personas")
    for p in problems:
        print("PROBLEM:", p)
    if args.check:
        return
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    OUTPUT.write_text(text, encoding="utf-8")
    print("Wrote", OUTPUT.relative_to(ROOT), "sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
