"""Text-only occupation-context/task treatment; frozen description controls stay intact."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from ..data_loader import read_csv
from ..models.description_reranker import DescriptionReranker, VARIANTS
from ..models.sequential_description_config import SEQUENTIAL_PARAMETERS, REVERSE_PARAMETERS
from .aggregation_config import AGGREGATED_PARAMETERS

TEXT_POLICY = {
    "source": "occupation_description_as_context_plus_all_tasks",
    "tasks_path": "data/tasks.csv",
    "chunk_tokens": 224,
    "chunk_overlap": 0,
    "document_aggregation": "mean_chunk_cosine",
    "numeric_occupation_inputs": "none",
    "precomputed_embeddings": False,
}


def treatment(parameters):
    parameters = deepcopy(parameters)
    parameters["variant"] += "_context_tasks_v1"
    parameters["text_policy"] = deepcopy(TEXT_POLICY)
    return parameters


CONFIGURATIONS = {
    "minilm_groups5": treatment(VARIANTS["blocks5_v1"]),
    "minilm_groups10": treatment(VARIANTS["blocks10_v1"]),
    "mpnet_groups10": treatment(VARIANTS["mpnet_blocks10_v1"]),
    "mpnet20_then_minilm10": treatment(SEQUENTIAL_PARAMETERS),
    "minilm20_then_mpnet10": treatment(REVERSE_PARAMETERS),
    "minilm_mpnet_equal_weights_groups10": treatment(AGGREGATED_PARAMETERS),
}


class ContextTaskReranker(DescriptionReranker):
    """Reuse candidate/group contracts; replace only the text score provider.

    RIASEC values, classifier outputs, context ratings and relevance labels never
    enter this object. Embeddings are generated from raw text during this run.
    """

    def __init__(self, occupations_path, eligible_codes, parameters, encoder=None):
        super().__init__(occupations_path, eligible_codes, encoder=encoder, parameters=parameters)
        self.source_hashes = {str(occupations_path): hashlib.sha256(Path(occupations_path).read_bytes()).hexdigest(),
                              TEXT_POLICY["tasks_path"]: hashlib.sha256(Path(TEXT_POLICY["tasks_path"]).read_bytes()).hexdigest()}
        tasks = {code: {} for code in eligible_codes}
        for row in read_csv(Path(TEXT_POLICY["tasks_path"]), {"code", "Task ID", "Task"}):
            if row["code"] in tasks:
                task_id, text = row["Task ID"], row["Task"].strip()
                if not text or task_id in tasks[row["code"]]:
                    raise ValueError("Empty or duplicate occupation task")
                tasks[row["code"]][task_id] = text
        if not all(tasks.values()):
            raise ValueError("Every eligible occupation requires tasks")
        self.descriptions = {code: "Context:\n" + description + "\nTasks:\n" + "\n".join(
            text for _, text in sorted(tasks[code].items())) for code, description in self.descriptions.items()}
        self.chunk_counts = {}
        self.model_files = {}

    def _load_encoder(self):
        if self.encoder is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)
        torch.manual_seed(425)
        torch.use_deterministic_algorithms(True)
        name = "MINILM" if self.parameters["dimension"] == 384 else "MPNET"
        local = os.environ.get(f"CONTEXT_TASK_{name}_MODEL")
        if local:
            model_path = Path(local)
            self.model_files = {p.relative_to(model_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(model_path.rglob("*")) if p.is_file() and ".cache" not in p.parts}
        self.encoder = SentenceTransformer(local or self.parameters["model_id"],
            **({} if local else {"revision": self.parameters["model_revision"]}),
            device="cpu", trust_remote_code=False, local_files_only=True)
        self.encoder.max_seq_length = self.parameters["max_seq_length"]
        self.encoder.eval()

    def _chunks(self, text):
        self._load_encoder()
        tokenizer = self.encoder.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = [tokenizer.decode(tokens[start:start + TEXT_POLICY["chunk_tokens"]],
                                   skip_special_tokens=True) for start in range(0, len(tokens), TEXT_POLICY["chunk_tokens"])]
        if any(len(tokenizer.encode(chunk)) > self.parameters["max_seq_length"] for chunk in chunks):
            raise ValueError("Chunk would be truncated by encoder")
        return chunks

    def score(self, codes, utterances):
        import numpy as np
        codes = tuple(codes)
        if len(codes) != len(set(codes)) or any(c not in self.descriptions for c in codes):
            raise ValueError("Semantic candidates must be unique eligible occupation codes")
        if len(codes) > self.parameters["candidate_count"]:
            raise ValueError("Semantic candidate count exceeds configured shortlist")
        query = "\n".join(text.strip() for text in utterances if text.strip())
        if not codes or not query:
            return {}
        missing = [code for code in codes if code not in self.cache]
        if missing:
            chunks = [self._chunks(self.descriptions[code]) for code in missing]
            vectors = self._encode([text for group in chunks for text in group])
            offset = 0
            for code, group in zip(missing, chunks):
                self.cache[code] = np.mean(vectors[offset:offset + len(group)], axis=0)
                self.chunk_counts[code] = len(group)
                offset += len(group)
        # Preserve the frozen query policy (newline join, model truncation at 256).
        vector = self._encode([query])[0]
        return {code: float(self.cache[code] @ vector) for code in codes}

    @property
    def metadata(self):
        return {**super().metadata, "text_policy": deepcopy(TEXT_POLICY),
                "source_sha256": self.source_hashes, "model_files_sha256": self.model_files,
                "encoded_candidate_chunk_counts": dict(sorted(self.chunk_counts.items())),
                "context_definition": "unclassified occupation Description prose; not Work Context ratings"}
