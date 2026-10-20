"""Conservative, text-only task stance extraction; no dataset annotations."""

from dataclasses import asdict, dataclass, replace
import re

REVISION = "explicit_scoped_v1"


@dataclass(frozen=True)
class TaskPreference:
    source_span: str
    activity_text: str
    polarity: str
    certainty: str
    status: str
    rule: str


class TaskPreferenceExtractor:
    """Explicit first-person stance only; omissions are unknown, never dislikes.

    Exact normalized activity contradictions are withheld. Paraphrased conflicts
    cannot reliably be resolved by these rules and remain a documented limitation.
    """

    revision = REVISION
    _split = re.compile(r"[.!?;\n]+|\b(?:but|however|yet|then again|although|because|since)\b|,\s*(?=(?:and\s+)?(?:I\b|i\b))", re.I)
    _uncertain = re.compile(r"\b(?:not sure|unsure|never tried|haven't tried|have not tried|might|maybe|perhaps|wouldn't mind|don't hate|do not hate|don't dislike|do not dislike)\b", re.I)
    _third = re.compile(r"\b(?:my (?:mum|mom|mother|dad|father|family|brother|sister|parents|uncle)|he|she|they)\b", re.I)
    _prefix = r"(?:\bI(?:'m| am)?\s+(?:(?:also|really|quite|absolutely|honestly|just|definitely|particularly)\s+)*)"
    _forward = re.compile(_prefix + r"(?P<cue>don't\s+(?:really\s+)?(?:enjoy|like|love)|do not\s+(?:enjoy|like|love)|dislike|hate|can't stand|cannot stand|am not into|not into|love|enjoy|like|prefer)(?:\s+)(?P<activity>.+)", re.I)
    _backward = re.compile(r"(?P<activity>.+?)\s+(?P<cue>is|are|feels?)\s+(?P<stance>(?:what I(?:'m| am) (?:most into|happiest doing)|what I (?:like|love|enjoy|care about) most|(?:easily )?my favou?rite (?:thing|part)|(?:really |so |kind of |kinda )?(?:fun|nice|satisfying|rewarding|great)))(?:\b|$)", re.I)
    _negative_tail = re.compile(r"(?P<activity>.+?)\s+(?:(?:is|are|isn't|aren't)\s+)?(?P<cue>(?:really |definitely )?not (?:really )?for me|no thanks|doesn't appeal to me|don't appeal to me|isn't for me|aren't for me)", re.I)

    @staticmethod
    def _clean(activity):
        activity = re.sub(r"^(?:and\s+)?(?:also\s+)?", "", activity.strip(), flags=re.I)
        activity = re.sub(r"^(?:most\s+)?is\s+", "", activity, flags=re.I)
        activity = re.split(r",\s*(?:because|since|so|I\b)|\b(?:because|though|to be fair)\b", activity, maxsplit=1, flags=re.I)[0]
        return activity.strip(" ,:;.!?")

    def extract(self, utterances):
        if not isinstance(utterances, (tuple, list)) or any(not isinstance(s, str) for s in utterances):
            raise ValueError("stance extractor accepts raw user utterances only")
        result = []
        for text in utterances:
            text = text.replace("\u2019", "'")
            for raw in self._split.split(text):
                span = raw.strip(" ,")
                if not span:
                    continue
                activity, polarity, rule = "", "unknown", "no_explicit_preference"
                certainty, status = "unknown", "unusable"
                if self._uncertain.search(span):
                    rule = "uncertain_or_negated_dislike"
                elif self._third.search(span):
                    rule = "third_party_or_background_scope"
                else:
                    scoped = span
                    if re.match(r"^(?:dislike|hate|enjoy|love|prefer)\s", span, re.I) and re.search(r"\bI\s", text, re.I):
                        scoped = "I " + span
                    match = self._forward.search(scoped)
                    if match:
                        activity = self._clean(match['activity'])
                        cue = match['cue'].lower()
                        polarity = "negative" if cue.startswith(("don't", "do not", "dislike", "hate", "can't", "cannot", "not", "am not")) else "positive"
                        rule = "explicit_first_person"
                    else:
                        match = self._backward.search(span)
                        if match:
                            activity, polarity, rule = self._clean(match['activity']), "positive", "activity_preference_predicate"
                        else:
                            match = self._negative_tail.search(span)
                            if match:
                                activity, polarity, rule = self._clean(match['activity']), "negative", "explicit_negative_predicate"
                    if activity:
                        # Do not embed anaphora, credentials, vague claims or a
                        # second stance whose scope was not safely segmented.
                        unsafe = re.search(r"\b(?:don't|not|never|dislike|hate|enjoy|love)\b", activity, re.I)
                        if len(activity.split()) < 2 or activity.lower() in {"it", "that", "this", "my job"} or unsafe:
                            polarity, rule = "unknown", "unclear_activity_scope"
                        else:
                            certainty, status = "explicit", "usable"
                result.append(TaskPreference(span, activity, polarity, certainty, status, rule))
        groups = {}
        for i, item in enumerate(result):
            if item.status == "usable":
                key = re.sub(r"[^a-z0-9 ]", "", item.activity_text.lower())
                groups.setdefault(key, []).append(i)
        for indices in groups.values():
            if len({result[i].polarity for i in indices}) > 1:
                for i in indices:
                    result[i] = replace(result[i], polarity="unknown", status="unresolved", rule="contradictory_same_activity")
        return result

    def trace(self, utterances):
        return [asdict(item) for item in self.extract(utterances)]
