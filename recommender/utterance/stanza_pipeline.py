"""Conservative Stanza meaning-unit baseline, preserving exact source spans."""
from pathlib import Path
from .pipeline import Clause, ProcessedUtterance

PACKAGES = {"tokenize": "combined", "mwt": "combined", "pos": "combined_nocharlm",
            "lemma": "combined_nocharlm", "depparse": "combined_nocharlm"}


class StanzaUtteranceProcessingPipeline:
    revision = "stanza_conservative_v1"

    def __init__(self, model_dir=None, *, nlp=None):
        self.model = self.revision
        if nlp is None:
            import stanza
            directory = Path(model_dir) if model_dir else Path(__file__).resolve().parents[2] / "models/stanza"
            try:
                nlp = stanza.Pipeline("en", dir=str(directory), processors=PACKAGES,
                                      package=None, download_method=None, use_gpu=False, verbose=False)
            except (OSError, ValueError) as exc:
                raise RuntimeError("Stanza models unavailable; run scripts/prepare_stanza.py first") from exc
        self.nlp = nlp

    def process(self, request):
        return tuple(self.split(text) for text in request.user_sentences)

    def split(self, text):
        if not isinstance(text, str):
            raise TypeError("utterance must be a string")
        if not text.strip():
            return ProcessedUtterance(text, ())
        clauses = []
        for sentence_index, sentence in enumerate(self.nlp(text).sentences):
            words = sentence.words
            by_id = {w.id: w for w in words}
            children = {w.id: [] for w in words}
            for word in words:
                if word.head in children:
                    children[word.head].append(word)
            roots = {w.id for w in words if w.head == 0}
            if not roots:
                continue
            sentence_root = next(iter(roots))
            # Split only independent coordinated/paratactic assertions with their
            # own subject. Never separate ccomp/xcomp/advcl/relative complements.
            # Shared-subject coordination stays together to preserve polarity scope.
            for word in words:
                if word.deprel.split(":")[0] not in {"conj", "parataxis"}:
                    continue
                if word.upos not in {"VERB", "AUX", "ADJ", "NOUN"}:
                    continue
                if word.head not in roots:
                    continue
                if any(c.deprel.split(":")[0] in {"nsubj", "csubj"} for c in children[word.id]):
                    roots.add(word.id)
            groups = {r: [] for r in roots}
            for word in words:
                owner = word
                seen = set()
                while owner.id not in roots:
                    if owner.id in seen or owner.head not in by_id:
                        owner = by_id[sentence_root]
                        break
                    seen.add(owner.id)
                    owner = by_id[owner.head]
                groups[owner.id].append(word)
            spans = []
            for root_id, group in groups.items():
                # Word offsets can be absent for expanded contractions. Token
                # offsets remain authoritative, so select through parent tokens.
                group_ids = {g.id for g in group}
                tokens = [t for t in sentence.tokens if any(w.id in group_ids for w in t.words)]
                while tokens and (all(w.upos == "PUNCT" for w in tokens[0].words)
                                  or all(w.deprel == "cc" for w in tokens[0].words)):
                    tokens.pop(0)
                while tokens and all(w.upos == "PUNCT" for w in tokens[-1].words):
                    tokens.pop()
                if tokens:
                    spans.append((tokens[0].start_char, tokens[-1].end_char, root_id))
            spans.sort()
            # A discontinuous subtree must not duplicate or omit source content.
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                tokens = list(sentence.tokens)
                while tokens and all(w.upos == "PUNCT" for w in tokens[-1].words):
                    tokens.pop()
                spans = [(tokens[0].start_char, tokens[-1].end_char, sentence_root)] if tokens else []
            for start, end, root_id in spans:
                word = by_id[root_id]
                parent = by_id.get(word.head)
                clauses.append(Clause(text[start:end], start, end, sentence_index, word.deprel,
                                      parent_start=getattr(parent, "start_char", None)))
        return ProcessedUtterance(text, tuple(clauses))
