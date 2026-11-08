"""Explainable initial rules; labels describe content, never preference polarity."""
from dataclasses import dataclass

# Deliberately excludes preference verbs and generic 'work'.
WORK_ACTIONS = frozenset("repair install maintain inspect build design develop program code analyse analyze research investigate test diagnose treat teach train tutor counsel advise care assist help manage lead supervise organise organize plan coordinate sell market negotiate persuade write edit illustrate paint compose perform calculate audit account record file process operate assemble manufacture cook clean transport deliver drive grow harvest protect enforce monitor evaluate assess prepare serve support create construct troubleshoot".split())
# Explicit phrase inventory: avoids matching 'independent' inside 'help people become independent'.
NATURE_PHRASES = (
    "noise", "noisy", "loud environment", "loud environments", "quiet environment",
    "quiet environments", "noise level", "noise levels", "high noise level",
    "high noise levels", "low noise level", "low noise levels",
    "outdoors", "outdoor work", "outdoor role", "outside work", "indoors", "indoor work",
    "in an office", "office based", "office-based", "in a laboratory", "remote work",
    "working alone", "work alone", "working independently", "work independently",
    "independent work", "independent role", "autonomous work", "structured routines",
    "routine work", "repetitive work", "physically demanding", "heavy lifting",
    "frequent travel", "travel frequently", "night shifts", "night shift", "flexible hours",
    "public interaction", "customer facing", "customer-facing", "team based", "team-based",
    "fast paced", "fast-paced", "high pressure", "high-pressure",
)


@dataclass(frozen=True)
class PatternEvidence:
    text: str
    start: int
    end: int
    rule: str


def classify(doc, start, end):
    tokens = [t for t in doc if start <= t.idx and t.idx + len(t) <= end]
    allowed = {t.i for t in tokens}
    activity = []
    for token in tokens:
        if token.pos_ != "VERB" or token.lemma_.lower() not in WORK_ACTIONS:
            continue
        targets = [c for c in token.children if c.i in allowed and c.dep_ in {"dobj", "obj", "dative", "nsubjpass"}]
        # Some parses attach the helped person as the subject of the complement.
        if token.lemma_.lower() in {"help", "assist"}:
            for complement in token.children:
                if complement.dep_ in {"xcomp", "ccomp"}:
                    targets.extend(c for c in complement.children
                                   if c.i in allowed and c.dep_ == "nsubj")
        for prep in token.children:
            if prep.i in allowed and prep.dep_ == "prep":
                targets.extend(c for c in prep.children if c.i in allowed and c.dep_ == "pobj")
        for target in targets:
            # Include intervening words and negation without inferring positive preference.
            members = [t for t in target.subtree if t.i in allowed]
            lo = min(token.idx, min(t.idx for t in members))
            hi = max(token.idx + len(token), max(t.idx + len(t) for t in members))
            activity.append(PatternEvidence(doc.text[lo:hi], lo, hi, "work_action_target"))
    # Multiword actions have a nominal core (take CARE of, look AFTER).
    # Do not add generic 'take'/'look' to the work-action vocabulary.
    for i, token in enumerate(tokens):
        lemmas = [t.lemma_.lower() for t in tokens[i:i + 3]]
        length = (3 if lemmas == ["take", "care", "of"] else
                  2 if lemmas[:2] == ["look", "after"] else 0)
        if not length:
            continue
        prep = tokens[i + length - 1]
        targets = [c for c in prep.children if c.i in allowed and c.dep_ in {"pobj", "obj"}]
        for target in targets:
            members = [t for t in target.subtree if t.i in allowed]
            hi = max(t.idx + len(t) for t in members)
            activity.append(PatternEvidence(doc.text[token.idx:hi], token.idx, hi,
                                            "caregiving_phrase_target"))
    nature = []
    words = [t.lower_ for t in tokens]
    for phrase in NATURE_PHRASES:
        parts = phrase.split()
        for i in range(len(words) - len(parts) + 1):
            if words[i:i + len(parts)] == parts:
                lo, last = tokens[i].idx, tokens[i + len(parts) - 1]
                hi = last.idx + len(last)
                nature.append(PatternEvidence(doc.text[lo:hi], lo, hi, "job_characteristic_phrase"))
    # Prefer the longest evidence span when vocabulary phrases overlap.
    nature = [e for e in nature if not any(
        other.start <= e.start and other.end >= e.end and
        (other.start < e.start or other.end > e.end) for other in nature)]
    return tuple(dict.fromkeys(activity)), tuple(dict.fromkeys(nature))
