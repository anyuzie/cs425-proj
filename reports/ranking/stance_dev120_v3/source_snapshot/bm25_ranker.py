"""M3 independent BM25 baseline over eligible prepared occupation documents."""

from collections import Counter, defaultdict
import math
import re

from .components import ComponentInput, ComponentRanking, ComponentScore
from .schemas import Occupation

# Frozen lexical-only preprocessing. Negators are deliberately retained.
STOPWORDS = tuple(sorted(set("""a an the and or but if then than as at by for from in into of on
to with is am are was were be been being do does did doing have has had having
i me my mine myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs themselves
this that these those there here what which who whom when where why how
can could would should will shall may might must also very really just so too""".split())))
BASELINE_PARAMETERS = {"k1": 1.5, "b": 0.75, "idf": "log1p((N-df+0.5)/(df+0.5))",
                       "tokenizer": "ascii_lower_alnum_apostrophe_v1",
                       "token_pattern": r"[a-z0-9]+(?:'[a-z]+)?", "stopwords": list(STOPWORDS),
                       "query_tf": "linear", "query": "all_user_utterances",
                       "document": "bm25_doc", "positive_matches_only": True,
                       "stemming": False, "tie_break": "code_ascending"}
_TOKEN = re.compile(BASELINE_PARAMETERS["token_pattern"])
_STOP = frozenset(STOPWORDS)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]


class BM25Ranker:
    def __init__(self, occupations: list[Occupation], parameters: dict):
        if parameters != BASELINE_PARAMETERS:
            raise ValueError("bm25@baseline_v1 requires its frozen parameters")
        if not occupations or len({o.code for o in occupations}) != len(occupations):
            raise ValueError("BM25 requires a nonempty unique eligible pool")
        self._postings = defaultdict(list)
        self._lengths = {}
        for occupation in sorted(occupations, key=lambda o: o.code):
            if not isinstance(occupation.bm25_doc, str) or not occupation.bm25_doc.strip():
                raise ValueError(f"{occupation.code}: missing eligible bm25_doc")
            terms = Counter(tokenize(occupation.bm25_doc))
            if not terms:
                raise ValueError(f"{occupation.code}: bm25_doc has no indexable tokens")
            self._lengths[occupation.code] = sum(terms.values())
            for term, count in terms.items():
                self._postings[term].append((occupation.code, count))
        self._average_length = sum(self._lengths.values()) / len(occupations)
        self._idf = {term: math.log1p((len(occupations) - len(postings) + 0.5) / (len(postings) + 0.5))
                     for term, postings in self._postings.items()}

    @property
    def metadata(self):
        return {"index": "in_memory_inverted_index", "documents": len(self._lengths),
                "vocabulary_size": len(self._postings), "average_document_length": self._average_length,
                "ranking_depth": "all strictly positive matches; no cutoff or zero-score padding"}

    def score(self, inputs: ComponentInput) -> ComponentRanking:
        # Accumulated wording, negation and repetition remain intact upstream.
        query = Counter(tokenize("\n".join(inputs.user_sentences)))
        if not query:
            return ComponentRanking("unusable", reason="BM25 query has no non-stopword tokens.")
        scores = defaultdict(float)
        k1, b = BASELINE_PARAMETERS["k1"], BASELINE_PARAMETERS["b"]
        # Sorting removes floating-point dependence on query token order.
        for term, query_count in sorted(query.items()):
            for code, tf in self._postings.get(term, ()):
                norm = k1 * (1 - b + b * self._lengths[code] / self._average_length)
                scores[code] += query_count * self._idf[term] * tf * (k1 + 1) / (tf + norm)
        matches = sorted(((code, score) for code, score in scores.items() if score > 0),
                         key=lambda item: (-item[1], item[0]))
        if not matches:
            return ComponentRanking("unusable", reason="BM25 query has no positive occupation matches.")
        return ComponentRanking("usable", tuple(ComponentScore(code, score, rank)
                                                 for rank, (code, score) in enumerate(matches, 1)))
