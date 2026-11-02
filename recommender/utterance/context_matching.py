"""Independent context relevance projection and explicit preference baseline."""
import csv
import hashlib
import re
from pathlib import Path
import numpy as np

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
MIN_CLAUSE_SCORE = 0.3


def below_clause_score_floor(prediction):
    return (prediction["task_score"] < MIN_CLAUSE_SCORE
            and prediction["context_score"] < MIN_CLAUSE_SCORE)


def preference(text):
    """Conservative lexical scope baseline; not general sentiment analysis."""
    target = text.strip().replace("’", "'")
    target = re.sub(r"\bdont\b", "do not", target, flags=re.I)
    target = re.sub(r"\b(do not|don't)\s+(?:really\s+)?(?=like|enjoy|want)", "do not ", target, flags=re.I)
    if re.search(r"\b(not sure|unsure|maybe|might|don't dislike|do not dislike|don't hate|do not hate|wouldn't mind)\b", target, re.I):
        return {"sign": None, "polarity": "unknown", "target": target, "rule": "uncertain_or_complex_negation"}
    negative = re.search(r"\b(dislike|hate|avoid|can't stand|cannot stand|don't (?:like|enjoy|want)|do not (?:like|enjoy|want)|prefer not to)\b", target, re.I)
    positive = re.search(r"\b(like|love|enjoy|prefer|want)\b", target, re.I)
    if negative and positive and positive.start() < negative.start():
        return {"sign": None, "polarity": "unknown", "target": target, "rule": "mixed_or_ambiguous_scope"}
    if negative:
        remainder = target[negative.end():].strip()
        if re.search(r"\b(but|although|however|like|love|enjoy|prefer|want)\b", remainder, re.I):
            return {"sign": None, "polarity": "unknown", "target": target, "rule": "mixed_or_ambiguous_scope"}
        return {"sign": -1, "polarity": "negative", "target": remainder or target, "rule": "explicit_negative"}
    if positive:
        remainder = target[positive.end():].strip()
        if re.search(r"\b(not|never|but|however)\b", remainder, re.I):
            return {"sign": None, "polarity": "unknown", "target": target, "rule": "ambiguous_scope"}
        return {"sign": 1, "polarity": "positive", "target": remainder or target, "rule": "explicit_positive"}
    if re.search(r"\b(not|never|no|doesn't|don't|isn't|can't)\b", target, re.I):
        return {"sign": None, "polarity": "unknown", "target": target, "rule": "unresolved_negation"}
    return {"sign": 1, "polarity": "positive", "target": target, "rule": "neutral_as_positive"}


class ContextMatchingPipeline:
    def __init__(self, dimensions_path, *, encoder=None, min_similarity=0.5):
        path = Path(dimensions_path)
        with path.open(encoding="utf-8-sig") as handle:
            self.dimensions = list(csv.DictReader(handle))
        ids = [d["context_id"] for d in self.dimensions]
        if len(ids) != 57 or len(set(ids)) != 57 or ids != sorted(ids):
            raise ValueError("Expected 57 unique sorted context dimensions")
        if [int(d["dimension"]) for d in self.dimensions] != list(range(1, 58)):
            raise ValueError("Context dimension indices must be 1 through 57")
        if not isinstance(min_similarity, (int, float)) or not np.isfinite(min_similarity) or not 0 < min_similarity <= 1:
            raise ValueError("Context similarity cutoff must be in (0, 1]")
        self.min_similarity = float(min_similarity)
        self.encoder = encoder
        self.attribute_vectors = None
        self.metadata = {"model_id": MODEL_ID, "revision": REVISION,
                         "dimensions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "dimension_ids": ids, "projection": "raw_cosine_v1",
                         "preference_rules": "explicit_lexical_v2", "neutral_policy": "positive",
                         "attribute_selection": "explicit_or_threshold_v2",
                         "min_similarity": self.min_similarity, "cutoff_calibrated": False}

    def _encode(self, texts):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu", trust_remote_code=False)
            self.encoder.eval()
        values = np.asarray(self.encoder.encode(texts, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(texts) or not np.isfinite(values).all():
            raise ValueError("Invalid context encoder output")
        if not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-4):
            raise ValueError("Context encoder must return normalized embeddings")
        return values

    def process(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Context clause must be nonempty text")
        stance = preference(text)
        if self.attribute_vectors is None:
            self.attribute_vectors = self._encode([d["context"] + ". " + d["description"] for d in self.dimensions])
        vector = np.clip(self.attribute_vectors @ self._encode([stance["target"]])[0], -1., 1.)
        # Resolve only explicit targets; never guess a pronoun's antecedent.
        unresolved = bool(re.search(r"\b(it|that|this|those|these|them|there)\b", stance["target"], re.I))
        # Expletive 'it' is safe when an explicit infinitival target follows.
        if re.match(r"it (?:is|would be) .+? to \w", stance["target"], re.I):
            unresolved = False
        matches = []
        status = "matched"
        if unresolved:
            status = "unresolved_reference"
        elif stance["sign"] is None:
            status = "unresolved_preference"
        else:
            explicit = []
            mappings = [
                (r"\b(?:office|air[- ]conditioned|climate[- ]controlled)\b", ["4.C.2.a.1.a"]),
                (r"\b(?:outdoors?|outside)\b", ["4.C.2.a.1.c", "4.C.2.a.1.d"]),
                (r"\b(?:indoors?|inside)\b", ["4.C.2.a.1.a", "4.C.2.a.1.b"]),
                (r"\b(?:noise|noisy|loud)\b", ["4.C.2.b.1.a"]),
            ]
            for pattern, ids in mappings:
                found = re.search(pattern, text, re.I)
                if found:
                    for key in ids:
                        explicit.append((key, found.group(), found.start(), found.end()))
            # Low-level conditions need a desired-level scorer, not a high-exposure list.
            unsupported = re.search(r"\b(quiet|remote|work from home|little noise|low noise|less noise|not outdoors|not outside)\b", text, re.I)
            if unsupported:
                status = "unsupported_desired_condition"
            elif explicit:
                seen = set()
                for key, phrase, start, end in explicit:
                    if key in seen:
                        continue
                    seen.add(key)
                    i = next(i for i, d in enumerate(self.dimensions) if d["context_id"] == key)
                    matches.append({"context_id": key, "context": self.dimensions[i]["context"],
                                    "similarity": float(max(vector[i], 0.)), "selection": "explicit_phrase",
                                    "source_phrase": phrase, "source_start": start, "source_end": end})
            else:
                for i in np.argsort(-vector, kind="stable"):
                    if vector[i] < self.min_similarity:
                        continue
                    matches.append({"context_id": self.dimensions[i]["context_id"],
                                    "context": self.dimensions[i]["context"], "similarity": float(vector[i]),
                                    "selection": "semantic_cutoff", "source_phrase": stance["target"]})
                    if len(matches) == 3:
                        break
            if status == "matched" and not matches:
                status = "no_relevant_attribute"
        return {"preference": stance, "context_vector": [float(v) for v in vector],
                "top_contexts": matches, "matching_status": status,
                "matching_usable": bool(matches)}



def route_clause_pipelines(text, prediction, *, task_pipeline=None, context_pipeline=None):
    """Independent gates; either pipeline may be absent. No occupation ranking."""
    if below_clause_score_floor(prediction):
        return {}
    results = {}
    if prediction["route_task"] and task_pipeline is not None:
        results["task"] = task_pipeline.process(text)
    if prediction["route_context"] and context_pipeline is not None:
        results["context"] = context_pipeline.process(text)
    return results
