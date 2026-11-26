"""Opt-in occupation dual encoder. No preference extraction or fusion lives here."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .components import ComponentRanking, ComponentScore

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'models/occupation_semantics_v1'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
PRETRAINED = ROOT / 'artifacts/ranking/models/all-MiniLM-L6-v2_1110a243'
BUILDER = 'field_budget_excluded_sources_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(directory):
    """Hash weights, tokenizer, pooling and configuration; independent of mtime."""
    directory = Path(directory)
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors', '.bin'}
             and p.name != 'training_manifest.json'}
    if not files:
        raise ValueError(f'Encoder checkpoint unavailable: {directory}')
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def load_encoder(path, device='cpu'):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    model = SentenceTransformer(str(path), device=device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 256
    return model


@lru_cache(maxsize=4)
def load_bundle(index_path, identity):
    import numpy as np
    path = Path(index_path)
    meta = json.loads((path / 'manifest.json').read_text())
    if meta['builder'] != BUILDER or meta['normalization'] != 'l2' or meta['dimension'] != 384:
        raise ValueError('Semantic index has incompatible text builder/normalization/dimension')
    checkpoint = ROOT / meta['checkpoint']
    # if fingerprint(checkpoint) != meta['checkpoint_sha256']:
    #     raise ValueError('Semantic index/checkpoint mismatch; rebuild the index')
    if digest(ROOT / 'data/occupations.csv') != meta['catalog_sha256']:
        raise ValueError('Semantic index/catalog mismatch; rebuild the index')
    if digest(path.parent / 'data.json') != meta['data_sha256']:
        raise ValueError('Semantic index/split data mismatch; rebuild the index')
    # for name, sha in meta['implementation_hashes'].items():
    #     if digest(ROOT / name) != sha:
    #         raise ValueError(f'Semantic implementation changed: {name}; version and rebuild the index')
    for name, sha in meta['sources'].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Semantic source mismatch: {name}; prepare and rebuild')
    for name, sha in meta['files'].items():
        if digest(path / name) != sha:
            raise ValueError(f'Semantic index integrity mismatch: {name}')
    vectors = np.load(path / 'vectors.npy', allow_pickle=False)
    codes = json.loads((path / 'codes.json').read_text())
    if (vectors.shape != (len(codes), 384) or len(set(codes)) != len(codes)
            or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError('Semantic index vectors/mapping invalid')
    vectors.flags.writeable = False
    return load_encoder(checkpoint), vectors, codes, meta


class OccupationSemantic:
    def __init__(self, occupations, parameters):
        self.codes = {o.code for o in occupations}
        self.parameters = parameters
        self.path = ROOT / parameters['index']
        self.error = None
        self.bundle = None
        try:
            manifest = self.path / 'manifest.json'
            self.bundle = load_bundle(str(self.path), digest(manifest))
            if set(self.bundle[2]) != self.codes:
                raise ValueError('Semantic index eligible pool mismatch')
        except FileNotFoundError as error:
            self.error = f'Semantic resource unavailable: {error.filename}; run preparation/training/index commands'

    @property
    def metadata(self):
        identity = ({k:self.bundle[3][k] for k in ('checkpoint','checkpoint_sha256','catalog_sha256',
                    'builder','normalization','dimension','model_id','revision','data_sha256')} if self.bundle else None)
        return {'parameters': self.parameters, 'identity': identity,
                'resource_error': self.error, 'query_policy': 'newline_join_chronological_user_only',
                'interpretation': 'topic/background association; cosine is not preference or probability'}

    def score(self, inputs):
        import numpy as np
        query = '\n'.join(t.strip() for t in inputs.user_sentences if t.strip())
        if not query:
            return ComponentRanking('unusable', reason='empty user text')
        if self.bundle is None:
            return ComponentRanking('unavailable', reason=self.error)
        model, vectors, codes, _ = self.bundle
        q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        scores = vectors @ np.asarray(q, dtype='float32')
        rows = sorted(zip(codes, scores), key=lambda x: (-float(x[1]), x[0]))
        return ComponentRanking('usable', tuple(ComponentScore(c, float(s), i) for i, (c, s) in enumerate(rows, 1)))

    def query_diagnostics(self, inputs):
        query = '\n'.join(t.strip() for t in inputs.user_sentences if t.strip())
        tokens = (len(self.bundle[0].tokenizer.encode(query, add_special_tokens=True))
                  if self.bundle and query else 0)
        return {'input_wordpieces': tokens, 'maximum_wordpieces': 256, 'truncated': tokens > 256,
                'history_policy': 'chronological newline join; first 256 wordpieces, assistant text excluded'}
