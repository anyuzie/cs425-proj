"""Task preference scoring; only user_sentences supplies evidence."""

from copy import deepcopy
from dataclasses import asdict
import numpy as np

from ..components import ComponentRanking, ComponentScore
from .dense_ranker import validated_vectors
from ..schemas import TaskEvidence
from .stance_config import PARAMETERS, POSITIVE_PARAMETERS
from .task_preferences import TaskPreferenceExtractor


class StanceTaskRanker:
    def __init__(self, tasks, embeddings, encoder, parameters, metadata=None):
        if parameters not in (PARAMETERS, POSITIVE_PARAMETERS):
            raise ValueError("unsupported stance parameters")
        import faiss
        self.tasks, self.encoder = tuple(tasks), encoder
        self.parameters = deepcopy(parameters)
        self.embeddings = validated_vectors(embeddings, len(tasks), np.asarray(embeddings).shape[1])
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        faiss.omp_set_num_threads(1)
        self.codes = sorted({t.code for t in tasks})
        self.indices = {c: [i for i, t in enumerate(tasks) if t.code == c] for c in self.codes}
        self.extractor = TaskPreferenceExtractor()
        self._metadata = deepcopy(metadata or {})
        self._metadata['stance'] = deepcopy(parameters)
        self.last_trace = {}

    @classmethod
    def from_cache(cls, occupations, parameters):
        # Shared validated resource loading only. Never invoke baseline.score().
        from .dense_config import BASELINE_PARAMETERS
        from .dense_ranker import DenseRanker
        resources = DenseRanker.from_cache(occupations, BASELINE_PARAMETERS)
        return cls(resources.tasks, resources.embeddings, resources.encoder,
                   parameters, resources.metadata)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def score(self, inputs):
        evidence = self.extractor.extract(inputs.user_sentences)
        usable = [e for e in evidence if e.status == 'usable' and
                  (e.polarity == 'positive' or self.parameters['include_negative'])]
        self.last_trace = {'extractor_revision': self.extractor.revision,
                           'evidence': [asdict(e) for e in evidence], 'top_matches': []}
        if not usable:
            return ComponentRanking('unusable', reason='No explicit usable task preferences for this variant')
        query = validated_vectors(self.encoder.encode([e.activity_text for e in usable]),
                                  len(usable), self.index.d)
        distances, ids = self.index.search(query, self.index.ntotal)
        similarities = np.empty((len(usable), len(self.tasks)), dtype='float32')
        for i in range(len(usable)):
            similarities[i, ids[i]] = np.maximum(0.0, distances[i])
        positive = [i for i, e in enumerate(usable) if e.polarity == 'positive']
        negative = [i for i, e in enumerate(usable) if e.polarity == 'negative']
        scores, matches = {}, {}
        for code in self.codes:
            task_indices = self.indices[code]
            maxima = similarities[:, task_indices].max(axis=1)
            pos = float(maxima[positive].astype('float64').mean()) if positive else 0.0
            neg = float(maxima[negative].astype('float64').mean()) if negative else 0.0
            scores[code] = pos - self.parameters['negative_weight'] * neg
            trace = []
            for i, e in enumerate(usable):
                winner = min(task_indices, key=lambda j: (-float(similarities[i, j]), self.tasks[j].task_id))
                task = self.tasks[winner]
                trace.append({'source_span': e.source_span, 'activity_text': e.activity_text,
                              'polarity': e.polarity, 'task_id': task.task_id,
                              'task_text': task.text, 'match': float(similarities[i, winner])})
            matches[code] = {'code': code, 'positive_score': pos, 'negative_score': neg,
                             'score': scores[code], 'matches': trace}
        if not any(np.any(similarities[i] > 0) for i in range(len(usable))):
            return ComponentRanking('unusable', reason='No positive task similarity for extracted preferences')
        ordered = sorted(scores, key=lambda c: (-scores[c], c))
        self.last_trace['top_matches'] = [matches[c] for c in ordered[:10]]
        # Generic UI matches stay provisional; polarity lives in explicit traces.
        provisional = []
        for code in ordered:
            best = sorted(matches[code]['matches'], key=lambda m: (-m['match'], m['task_id']))
            seen = set()
            for m in best:
                if m['task_id'] not in seen and len(seen) < 3:
                    seen.add(m['task_id'])
                    provisional.append(TaskEvidence(m['task_id'], code, m['task_text'], m['match'], 'provisional'))
        return ComponentRanking('usable', tuple(ComponentScore(c, scores[c], i)
                                for i, c in enumerate(ordered, 1)), evidence=tuple(provisional))
