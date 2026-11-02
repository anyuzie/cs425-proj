"""Conservative dependency-based splitting with lossless source provenance."""
from dataclasses import dataclass
from .classification import PatternEvidence, classify


@dataclass(frozen=True)
class Clause:
    text: str
    start: int
    end: int
    sentence_index: int
    relation: str
    inherited_subject: str | None = None
    parent_start: int | None = None
    activity_evidence: tuple[PatternEvidence, ...] = ()
    job_nature_evidence: tuple[PatternEvidence, ...] = ()

    @property
    def activity_detected(self):
        return bool(self.activity_evidence)

    @property
    def job_nature_detected(self):
        return bool(self.job_nature_evidence)

    @property
    def outcome(self):
        return {(True, False): "activity", (False, True): "job nature",
                (True, True): "mixed", (False, False): "uncertain"}[
                    self.activity_detected, self.job_nature_detected]


@dataclass(frozen=True)
class ProcessedUtterance:
    text: str
    clauses: tuple[Clause, ...]


class UtteranceProcessingPipeline:
    def __init__(self, model="en_core_web_sm", *, nlp=None):
        if nlp is None:
            try:
                import spacy
                nlp = spacy.load(model, disable=["ner"])
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Utterance processing requires requirements-utterance.txt and "
                    "the en_core_web_sm model; see docs/UTTERANCE_PROCESSING.md"
                ) from exc
        if "parser" not in nlp.pipe_names:
            raise ValueError("utterance processing requires a dependency parser")
        self.nlp = nlp
        self.model = model

    def process(self, request):
        """Read only user_sentences from the combined text/RIASEC input."""
        return tuple(self.split(text) for text in request.user_sentences)

    def split(self, text):
        if not isinstance(text, str):
            raise TypeError("utterance must be a string")
        if not text.strip():
            return ProcessedUtterance(text, ())
        doc = self.nlp(text)
        clauses = []
        for sentence_index, sentence in enumerate(doc.sents):
            roots = [sentence.root]
            for token in sentence:
                own_subject = any(c.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"}
                                  for c in token.children)
                # Keep complements such as 'want to teach' and relative clauses intact.
                # Subordinate clauses split only when they have their own subject.
                if ((token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}) or
                        (token.dep_ in {"advcl", "ccomp"} and own_subject
                         and token.head.lemma_ not in {"help", "assist"})):
                    roots.append(token)
            root_ids = {token.i for token in roots}
            groups = {token.i: [] for token in roots}
            for token in sentence:
                owner = token
                while owner.i not in root_ids:
                    owner = owner.head
                groups[owner.i].append(token)
            for root in roots:
                tokens = groups[root.i]
                # A discontinuous parent cannot be represented by one exact source span.
                # Preserve the sentence instead of dropping or fabricating words.
                while tokens and (tokens[0].is_punct or tokens[0].is_space or tokens[0].dep_ == "cc"):
                    tokens = tokens[1:]
                while tokens and (tokens[-1].is_punct or tokens[-1].is_space or tokens[-1].dep_ == "cc"):
                    tokens = tokens[:-1]
                ids = [t.i for t in tokens]
                if ids and max(ids) - min(ids) + 1 != len(ids):
                    groups = {sentence.root.i: list(sentence)}
                    roots = [sentence.root]
                    break
            for root in roots:
                tokens = groups[root.i]
                while tokens and (tokens[0].is_punct or tokens[0].is_space or tokens[0].dep_ == "cc"):
                    tokens = tokens[1:]
                while tokens and (tokens[-1].is_punct or tokens[-1].is_space or tokens[-1].dep_ == "cc"):
                    tokens = tokens[:-1]
                if not tokens:
                    continue
                start, end = tokens[0].idx, tokens[-1].idx + len(tokens[-1])
                inherited = None
                if root.dep_ == "conj" and not any(c.dep_ in {"nsubj", "nsubjpass", "csubj"} for c in root.children):
                    ancestor = root.head
                    while True:
                        subjects = [c for c in ancestor.children if c.dep_ in {"nsubj", "nsubjpass", "csubj"}]
                        if subjects:
                            subject = subjects[0]
                            inherited = doc[subject.left_edge.i:subject.right_edge.i + 1].text
                            break
                        if ancestor.head == ancestor:
                            break
                        ancestor = ancestor.head
                activity, nature = classify(doc, start, end)
                clauses.append(Clause(text[start:end], start, end, sentence_index,
                                      root.dep_, inherited,
                                      root.head.idx if root != sentence.root else None, activity, nature))
        return ProcessedUtterance(text, tuple(sorted(clauses, key=lambda c: c.start)))
