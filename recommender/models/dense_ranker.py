"""Exact semantic task retrieval; model/cache loaded once, never rebuilt per turn."""

from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import re

import faiss
import numpy as np

from ..components import ComponentInput, ComponentRanking, ComponentScore
from ..data_loader import read_csv
from .dense_config import BASELINE_PARAMETERS
from ..schemas import TaskEvidence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_versions():
    return {name: version(name) for name in ("numpy", "torch", "transformers", "sentence-transformers", "faiss-cpu")}


@dataclass(frozen=True)
class Task:
    code: str
    task_id: str
    text: str


def load_tasks(path, eligible_codes):
    tasks, seen = [], set()
    for row in read_csv(Path(path), {"code", "Task ID", "Task"}):
        if row["code"] not in eligible_codes:
            continue
        key = (row["code"], row["Task ID"])
        if not row["Task ID"].strip() or not row["Task"].strip() or key in seen:
            raise ValueError(f"invalid/duplicate eligible task: {key}")
        seen.add(key)
        tasks.append(Task(row["code"], row["Task ID"], row["Task"]))
    if not tasks:
        raise ValueError("no eligible tasks for dense retrieval")
    return sorted(tasks, key=lambda task: (task.code, task.task_id))


def sentences(utterances):
    # Natural text, preserving negation and repetition; no BM25 tokenization.
    return [match.group().strip() for text in utterances for match in re.finditer(r"[^.!?\n]+[.!?]*", text)
            if any(char.isalnum() for char in match.group())]


def validated_vectors(values, count, dimension):
    vectors = np.asarray(values, dtype="float32")
    if vectors.shape != (count, dimension) or not np.isfinite(vectors).all():
        raise ValueError("invalid embedding shape or nonfinite embedding")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError("dense embeddings must be normalized unit vectors")
    return np.ascontiguousarray(vectors)


def model_identity(parameters):
    directory = Path(parameters["model_dir"])
    source = directory / "model_source.json"
    if not source.is_file():
        raise ValueError("pinned MiniLM snapshot missing; run python -m recommender.build_dense --download-model")
    metadata = json.loads(source.read_text(encoding="utf-8"))
    if (metadata["model_id"], metadata["revision"]) != (parameters["model_id"], parameters["model_revision"]):
        raise ValueError("MiniLM snapshot revision does not match variant")
    for name, sha in metadata["files"].items():
        if digest(directory / name) != sha:
            raise ValueError(f"MiniLM model file fingerprint mismatch: {name}")
    return fingerprint(metadata)


class MiniLMEncoder:
    def __init__(self, parameters):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(parameters["threads"])
        torch.manual_seed(parameters["seed"])
        torch.use_deterministic_algorithms(True)
        self.model = SentenceTransformer(parameters["model_dir"], device=parameters["device"],
                                         local_files_only=True, trust_remote_code=False)
        self.model.max_seq_length = parameters["max_seq_length"]
        self.model.eval()
        self.parameters = parameters

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.parameters["batch_size"],
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def cache_identity(occupations, parameters, tasks, model_hash):
    return {"format": "m4-flat-v1", "parameters": parameters, "packages": package_versions(),
            "pool": sorted(o.code for o in occupations), "tasks_sha256": digest(parameters["tasks_path"]),
            "task_mapping_sha256": fingerprint([task.__dict__ for task in tasks]),
            "model_snapshot_sha256": model_hash}


class DenseRanker:
    def __init__(self, occupations, tasks, embeddings, encoder, *, metadata=None):
        self._pool = {o.code for o in occupations}
        if not tasks or len({(t.code, t.task_id) for t in tasks}) != len(tasks) or any(t.code not in self._pool for t in tasks):
            raise ValueError("dense tasks must be unique and map to eligible occupations")
        self.tasks, self.encoder = tuple(tasks), encoder
        if np.asarray(embeddings).ndim != 2:
            raise ValueError("dense embeddings must be a matrix")
        self.embeddings = validated_vectors(embeddings, len(tasks), np.asarray(embeddings).shape[1])
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        faiss.omp_set_num_threads(1)
        self._codes = sorted({task.code for task in tasks})  # Missing-task occupations never receive ranks.
        indices = {code: i for i, code in enumerate(self._codes)}
        self._owners = np.array([indices[task.code] for task in tasks])
        self._task_indices = {code: np.flatnonzero(self._owners == indices[code]).tolist() for code in self._codes}
        self._metadata = metadata or {}

    @classmethod
    def from_cache(cls, occupations, parameters):
        if parameters != BASELINE_PARAMETERS:
            raise ValueError("minilm@mean_tasks_v1 requires its frozen parameters")
        tasks = load_tasks(parameters["tasks_path"], {o.code for o in occupations})
        expected = cache_identity(occupations, parameters, tasks, model_identity(parameters))
        directory = Path(parameters["cache_dir"])
        if not (directory / "manifest.json").is_file():
            raise ValueError("M4 task cache missing; run python -m recommender.build_dense")
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if metadata["identity"] != expected:
            raise ValueError("stale M4 cache: configuration, corpus, model, pool or dependency version changed")
        for name, sha in metadata["artifact_sha256"].items():
            if digest(directory / name) != sha:
                raise ValueError(f"M4 cache artifact fingerprint mismatch: {name}")
        owners = json.loads((directory / "owners.json").read_text(encoding="utf-8"))
        if owners != [task.__dict__ for task in tasks]:
            raise ValueError("M4 task owner mapping mismatch")
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        embeddings = validated_vectors(embeddings, len(tasks), parameters["dimension"])
        index = faiss.read_index(str(directory / "index.faiss"))
        if not isinstance(index, faiss.IndexFlatIP) or index.ntotal != len(tasks) or index.d != parameters["dimension"]:
            raise ValueError("invalid M4 FAISS index shape/type")
        if not np.array_equal(index.reconstruct_n(0, index.ntotal), embeddings):
            raise ValueError("M4 index and embedding cache disagree")
        result = cls(occupations, tasks, embeddings, MiniLMEncoder(parameters), metadata=metadata)
        result.index = index
        return result

    @property
    def metadata(self):
        return self._metadata

    def score(self, inputs: ComponentInput) -> ComponentRanking:
        texts = sentences(inputs.user_sentences)
        if not texts:
            return ComponentRanking("unusable", reason="No nonempty user sentences for dense retrieval.")
        query = validated_vectors(self.encoder.encode(texts), len(texts), self.index.d)
        distances, task_ids = self.index.search(query, self.index.ntotal)  # Exhaustive: no task cutoff bias.
        by_task = np.empty((len(texts), len(self.tasks)), dtype="float32")
        for i, (scores, ids) in enumerate(zip(distances, task_ids)):
            by_task[i, ids] = scores
        maximum = np.full((len(texts), len(self._codes)), -np.inf, dtype="float32")
        for i in range(len(texts)):
            np.maximum.at(maximum[i], self._owners, by_task[i])
        scores = maximum.astype("float64").mean(axis=0)
        ordered = sorted(zip(self._codes, scores), key=lambda item: (-item[1], item[0]))
        # Task matches are provisional, including when wording expresses a dislike.
        evidence = []
        task_scores = by_task.max(axis=0)
        for code in self._codes:
            matches = sorted(self._task_indices[code],
                             key=lambda i: (-float(task_scores[i]), self.tasks[i].task_id))[:3]
            evidence.extend(TaskEvidence(self.tasks[i].task_id, code, self.tasks[i].text,
                                         float(task_scores[i]), "provisional") for i in matches)
        return ComponentRanking("usable", tuple(ComponentScore(code, float(score), rank)
                                                 for rank, (code, score) in enumerate(ordered, 1)),
                                evidence=tuple(evidence))
