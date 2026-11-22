"""Sentence-to-task nearest-neighbour reranking.

Idea: a single strong match between one user sentence and one occupation task
statement is better evidence than the mean similarity between the whole user
history and a whole occupation document. Scoring reuses the ``minilm@mean_tasks_v1``
formula (per sentence, maximum task cosine within an occupation; mean over
sentences), but applies it as a second-stage reranker inside the weighted
dot-product groups of 10.

The signed treatment differs in one way: sentences that the conservative lexical
preference rule labels explicitly negative contribute ``-max_cosine``. Unknown and positive sentences are unchanged.
"""
from copy import deepcopy
import os
from pathlib import Path
import re

from ..models.description_reranker import VARIANTS
from .context_tasks import ContextTaskReranker

SENTENCE_PATTERN = r"[^.!?\n]+[.!?]*"

BASE_POLICY = {
    "source": "all_raw_occupation_tasks",
    "tasks_path": "data/tasks.csv",
    "sentence_splitter": "m4_regex_v1",
    "sentence_pattern": SENTENCE_PATTERN,
    "occupation_aggregation": "max_task_cosine_per_sentence",
    "query_aggregation": "mean_over_sentences",
    "preference": "none",
    "numeric_occupation_inputs": "none",
    "precomputed_embeddings": False,
}
SIGNED_POLICY = {**BASE_POLICY, "preference": "explicit_negative_sign_v1"}


LOCAL_MODEL_DEFAULTS = {
    "CONTEXT_TASK_MINILM_MODEL": Path("artifacts/ranking/models/all-MiniLM-L6-v2_1110a243"),
    "CONTEXT_TASK_MPNET_MODEL": Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2"
                                / "snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130",
}


def configure_local_models():
    """Point the pinned encoders at existing local snapshots unless already set.

    These paths only locate resources; their file hashes are recorded in run metadata.
    """
    for name, path in LOCAL_MODEL_DEFAULTS.items():
        if not os.environ.get(name) and path.is_dir():
            os.environ[name] = str(path)
    return {name: os.environ.get(name) for name in LOCAL_MODEL_DEFAULTS}


def split_sentences(utterances):
    """Same splitter as the dense ranker (copied to avoid importing FAISS)."""
    return [match.group().strip() for text in utterances for match in re.finditer(SENTENCE_PATTERN, text)
            if any(char.isalnum() for char in match.group())]


def sentence_sign(text, policy):
    if policy["preference"] == "none":
        return 1
    from ..utterance.context_matching import preference
    # Only explicit negatives flip; unknown/uncertain stays +1 (unsigned behaviour).
    return -1 if preference(text)["sign"] == -1 else 1


def treatment(base_key, policy, suffix):
    parameters = deepcopy(VARIANTS[base_key])
    parameters["variant"] += suffix
    parameters["text_policy"] = deepcopy(policy)
    return parameters


CONFIGURATIONS = {
    "minilm_groups10": treatment("blocks10_v1", BASE_POLICY, "_clause_tasks_v1"),
    "mpnet_groups10": treatment("mpnet_blocks10_v1", BASE_POLICY, "_clause_tasks_v1"),
    "minilm_groups10_signed": treatment("blocks10_v1", SIGNED_POLICY, "_clause_tasks_signed_v1"),
    "mpnet_groups10_signed": treatment("mpnet_blocks10_v1", SIGNED_POLICY, "_clause_tasks_signed_v1"),
}


def base_variant(parameters):
    """Encoder/group variant underlying a clause-task treatment."""
    return next(value for value in VARIANTS.values()
                if parameters["variant"].startswith(value["variant"] + "_clause_tasks"))


class ClauseTaskReranker(ContextTaskReranker):
    """Scores candidates from individual task statements; never reads RIASEC values.

    Only occupation codes, raw task text and user text reach this object. Task
    vectors are encoded at runtime and kept in memory for the process lifetime.
    """

    def __init__(self, occupations_path, eligible_codes, parameters, policy, encoder=None):
        if policy not in (BASE_POLICY, SIGNED_POLICY):
            raise ValueError("Unknown frozen clause-task policy")
        super().__init__(occupations_path, eligible_codes, parameters, encoder=encoder)
        self.policy = deepcopy(policy)
        self.task_cache = {}
        self.last_matches = {}

    def _task_matrix(self, codes):
        missing = [code for code in codes if code not in self.task_cache]
        if missing:
            texts = [text for code in missing for text in self.tasks[code].values()]
            vectors = self._encode(texts)
            offset = 0
            for code in missing:
                count = len(self.tasks[code])
                self.task_cache[code] = vectors[offset:offset + count]
                offset += count

    def score(self, codes, utterances):
        import numpy as np
        codes = tuple(codes)
        if len(codes) != len(set(codes)) or any(c not in self.tasks for c in codes):
            raise ValueError("Semantic candidates must be unique eligible occupation codes")
        if len(codes) > self.parameters["candidate_count"]:
            raise ValueError("Semantic candidate count exceeds configured shortlist")
        self.last_matches = {}
        sentences = split_sentences(utterances)
        if not codes or not sentences:
            return {}
        self._task_matrix(codes)
        queries = self._encode(sentences)
        signs = np.array([sentence_sign(text, self.policy) for text in sentences], dtype="float32")
        scores = {}
        for code in codes:
            similarities = self.task_cache[code] @ queries.T  # tasks x sentences
            best = similarities.argmax(axis=0)  # first maximum = lowest sorted task ID
            maxima = similarities[best, range(len(sentences))]
            scores[code] = float(np.mean(signs * maxima))
            task_ids = list(self.tasks[code])
            self.last_matches[code] = [
                {"sentence": text, "sign": int(sign), "task_id": task_ids[index],
                 "task_text": self.tasks[code][task_ids[index]], "cosine": float(value)}
                for text, sign, index, value in zip(sentences, signs, best, maxima)]
        return scores

    @property
    def metadata(self):
        values = super().metadata
        values["text_policy"] = deepcopy(self.policy)
        values["encoded_task_counts"] = {code: len(matrix) for code, matrix in sorted(self.task_cache.items())}
        values.pop("encoded_candidate_chunk_counts", None)
        values["context_definition"] = "raw O*NET task statements only; no description, ratings or classifier labels"
        return values
