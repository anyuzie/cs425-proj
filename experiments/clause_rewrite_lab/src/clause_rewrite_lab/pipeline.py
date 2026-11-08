"""The reconstruction branch evaluates rewrites; it never supplies matching text."""
from dataclasses import asdict, dataclass
import math
import re
from time import perf_counter


def features(text):
    value = text.casefold().replace("’", "'")
    uncertain = bool(re.search(r"\b(maybe|might|perhaps|unsure|not sure|wouldn't mind|don't mind|do not mind|don't dislike|don't hate)\b", value))
    negative = bool(re.search(r"\b(dislikes?|hates?|avoids?|nope|don't|doesn't|do not|does not|not enjoy|not like|cannot stand|can't stand)\b", value))
    positive = bool(re.search(r"\b(enjoys?|likes?|loves?|prefers?|passion)\b", value))
    polarity = "uncertain" if uncertain else "negative" if negative else "positive" if positive else "unspecified"
    frequency = set()
    for category, pattern in {"occasional": r"\b(sometimes|occasionally|occasionally|now and then)\b",
                              "frequent": r"\b(often|frequently|regularly)\b", "always": r"\b(always|every time)\b",
                              "never": r"\bnever\b"}.items():
        if re.search(pattern, value):
            frequency.add(category)
    return {"polarity": polarity, "frequency": sorted(frequency),
            "quantities": re.findall(r"\b\d+(?:\.\d+)?\b", value),
            "conditional": bool(re.search(r"\b(if|unless|provided that)\b", value)),
            "comparison": bool(re.search(r"\b(rather than|instead of|more than|less than|over)\b", value)),
            "strong_preference": bool(re.search(r"\b(loves?|hates?|passion|cannot stand|can't stand)\b", value))}


def invariant_reasons(original, candidate):
    left, right = features(original), features(candidate)
    reasons = []
    for key in left:
        if left[key] != right[key]:
            reasons.append(f"changed_{key}")
    if not candidate.strip():
        reasons.append("empty_rewrite")
    if "?" in candidate:
        reasons.append("question_instead_of_statement")
    if len(candidate.split()) > 64:
        reasons.append("rewrite_not_short")
    return reasons


@dataclass
class Candidate:
    text: str
    seed: int
    reconstruction: str
    direct_cosine: float
    cycle_cosine: float
    entailment: dict
    accepted: bool
    rejection_reasons: list[str]


class ClausePipeline:
    def __init__(self, generator, reconstructor, evaluator, matcher):
        self.generator = generator
        self.reconstructor = reconstructor
        self.evaluator = evaluator
        self.matcher = matcher

    def process(self, original, top_k=5):
        if not isinstance(original, str) or not original.strip():
            raise ValueError("Enter a nonempty clause")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        started = perf_counter()
        candidates, errors = [], []
        try:
            generated = self.generator.generate(original)
        except ValueError as error:
            generated, errors = [], [str(error)]
        for text, seed in generated:
            # Deliberately pass only the forward output to the reverse generator.
            recovered = self.reconstructor.reconstruct(text)
            direct, cycle, entailment = self.evaluator.evaluate(original, text, recovered)
            if not all(math.isfinite(value) for value in (direct, cycle, *entailment.values())):
                raise ValueError("Fidelity evaluator returned nonfinite values")
            reasons = invariant_reasons(original, text)
            reasons.extend("reconstruction_" + reason for reason in invariant_reasons(original, recovered)
                           if reason not in {"question_instead_of_statement", "rewrite_not_short"})
            if direct < .8:
                reasons.append("low_direct_similarity")
            if cycle < .8:
                reasons.append("low_cycle_similarity")
            if any(value < .5 for value in entailment.values()):
                reasons.append("meaning_not_bidirectionally_entailed")
            candidates.append(Candidate(text, seed, recovered, direct, cycle, entailment, not reasons, reasons))
        ranked = sorted(candidates, key=lambda c: (-(c.direct_cosine + c.cycle_cosine) / 2, c.seed))
        accepted = next((candidate for candidate in ranked if candidate.accepted), None)
        chosen = accepted or (ranked[0] if ranked else None)
        matching_text = accepted.text if accepted else original
        # No reconstruction text can reach either matcher call.
        matches = self.matcher.match(matching_text, top_k)
        preview = self.matcher.match(chosen.text, top_k) if chosen and not accepted else []
        return {"schema_version": 1, "original_clause": original,
                "diffuseq_rewrite": chosen.text if chosen else None,
                "rewrite_is_copy": bool(chosen and re.findall(r"\w+", chosen.text.casefold()) == re.findall(r"\w+", original.casefold())),
                "reconstruction": chosen.reconstruction if chosen else None,
                "rewrite_accepted": accepted is not None,
                "matching_text": matching_text, "matching_source": "diffuseq" if accepted else "original_fallback",
                "fidelity": asdict(chosen) if chosen else None,
                "candidates": [asdict(candidate) for candidate in candidates], "generation_errors": errors,
                "task_matches": matches, "rejected_rewrite_task_preview": preview,
                "preference": features(original)["polarity"],
                "interpretation": "semantic task associations; a negative preference indicates potential conflicts",
                "seconds": perf_counter() - started}
