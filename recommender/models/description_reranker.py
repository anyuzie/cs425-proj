"""Second-stage description-only embedding reranking; never retrieves candidates."""
from importlib.metadata import version
import hashlib
import json
import os

from ..data_loader import read_csv
from ..schemas import SignalScore
from .mpnet_description_config import MPNET_PARAMETERS, MPNET20_PARAMETERS

PARAMETERS = {
    "algorithm": "minilm_descriptions", "variant": "top100_v1", "candidate_count": 100,
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "dimension": 384, "max_seq_length": 256,
    "query": "newline_join_user_history", "ordering": "cosine_desc_code_asc",
}
VARIANTS = {"top100_v1": PARAMETERS, **{
    f"blocks{size}_v1": {**PARAMETERS, "variant": f"blocks{size}_v1", "block_size": size}
    for size in (5, 10, 20)}}
VARIANTS[MPNET_PARAMETERS["variant"]] = MPNET_PARAMETERS
VARIANTS[MPNET20_PARAMETERS["variant"]] = MPNET20_PARAMETERS


class DescriptionReranker:
    def __init__(self, occupations_path, eligible_codes, encoder=None, parameters=None):
        self.parameters = dict(PARAMETERS if parameters is None else parameters)
        if self.parameters not in VARIANTS.values():
            raise ValueError("Unknown frozen description reranking variant")
        self.descriptions = {r["code"]: r["Description"].strip()
                             for r in read_csv(occupations_path, {"code", "Description"})
                             if r["code"] in eligible_codes}
        if set(self.descriptions) != eligible_codes or not all(self.descriptions.values()):
            raise ValueError("Every eligible occupation requires a description")
        self.encoder = encoder
        self.cache = {}

    @property
    def metadata(self):
        corpus = json.dumps(self.descriptions, sort_keys=True).encode()
        return {"parameters": self.parameters, "descriptions_sha256": hashlib.sha256(corpus).hexdigest(),
                "packages": {p: version(p) for p in ("numpy", "torch", "sentence-transformers")}}

    def _encode(self, texts):
        import numpy as np
        if self.encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(1)
            torch.manual_seed(425)
            torch.use_deterministic_algorithms(True)
            self.encoder = SentenceTransformer(self.parameters["model_id"],
                revision=self.parameters["model_revision"], device="cpu", trust_remote_code=False,
                cache_folder=os.environ.get("RECOMMENDER_MODEL_CACHE", "artifacts/ranking/models"))
            self.encoder.max_seq_length = self.parameters["max_seq_length"]
            self.encoder.eval()
        values = np.asarray(self.encoder.encode(texts, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False), dtype="float32")
        if (values.shape != (len(texts), self.parameters["dimension"]) or
                not np.isfinite(values).all() or
                not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5)):
            raise ValueError(f"Encoder must return finite normalized {self.parameters['dimension']}-dimensional vectors")
        return values

    def score(self, codes, utterances):
        """Return description cosine scores without reading RIASEC or changing order."""
        codes = tuple(codes)
        if len(codes) != len(set(codes)) or any(c not in self.descriptions for c in codes):
            raise ValueError("Semantic candidates must be unique eligible occupation codes")
        if len(codes) > self.parameters["candidate_count"]:
            raise ValueError("Semantic candidate count exceeds configured shortlist")
        query = "\n".join(s.strip() for s in utterances if s.strip())
        if not codes or not query:
            return {}
        missing = [code for code in codes if code not in self.cache]
        if missing:
            self.cache.update(zip(missing, self._encode([self.descriptions[c] for c in missing])))
        vector = self._encode([query])[0]
        return {code: float(self.cache[code] @ vector) for code in codes}

    def rerank(self, response, utterances, *, signal_name="semantic"):
        if response.status != "ranked":
            return response
        candidates = response.occupations[:self.parameters["candidate_count"]]
        scores = self.score([o.code for o in candidates], utterances)
        if not scores:
            return response
        block_size = self.parameters.get("block_size", 100)
        candidates = [occupation for start in range(0, len(candidates), block_size)
                      for occupation in sorted(candidates[start:start + block_size],
                                               key=lambda o: (-scores[o.code], o.code))]
        semantic_ranks = {o.code: rank for rank, o in enumerate(
            sorted(candidates, key=lambda o: (-scores[o.code], o.code)), 1)}
        for rank, occupation in enumerate(candidates, 1):
            occupation.signals[signal_name] = SignalScore(scores[occupation.code], semantic_ranks[occupation.code])
        # Keep the untouched tail for full-pool MRR and fair baseline comparison.
        response.occupations = candidates + response.occupations[len(candidates):]
        for rank, occupation in enumerate(response.occupations, 1):
            occupation.rank = rank
        return response
