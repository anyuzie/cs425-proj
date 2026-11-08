"""Human annotation contracts and training-safe validation."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Span(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Clause(Span):
    edited_text: str | None = None
    uncertain: bool = False
    activity: bool | None = None
    job_nature: bool | None = None
    activity_evidence: list[Span] = Field(default_factory=list)
    job_nature_evidence: list[Span] = Field(default_factory=list)
    note: str = ""


class Annotation(StrictModel):
    workflow: Literal["span_review", "clause_labels"] = "span_review"
    annotator: str = Field(min_length=1, max_length=80)
    status: Literal["draft", "needs_review", "reviewed"]
    source_sha256: str
    expected_revision: int = Field(ge=0)
    group_id: str = Field(min_length=1, max_length=120)
    clauses: list[Clause]
    note: str = ""


def outcome(activity, nature):
    if activity is None or nature is None:
        return "unresolved"
    return {(True, False): "activity", (False, True): "job_nature",
            (True, True): "mixed", (False, False): "neither"}[activity, nature]


def validate_annotation(annotation, text):
    if not annotation.annotator.strip() or not annotation.group_id.strip():
        raise ValueError("Annotator and group ID cannot be blank")
    if annotation.workflow == "clause_labels":
        if not annotation.clauses:
            raise ValueError("Select and label at least one clause before saving")
        for clause in annotation.clauses:
            if not 0 <= clause.start < clause.end <= len(text):
                raise ValueError("A clause selection is outside the original utterance")
            value = clause.edited_text if clause.edited_text is not None else text[clause.start:clause.end]
            if not value.strip():
                raise ValueError("Clause text cannot be empty")
            if clause.uncertain:
                if clause.activity is not None or clause.job_nature is not None:
                    raise ValueError("Uncertain clauses cannot also have activity or job-nature labels")
            elif clause.activity is None or clause.job_nature is None or not (clause.activity or clause.job_nature):
                raise ValueError("Choose Activity, Job nature, both, or Uncertain for every clause")
            if clause.activity_evidence or clause.job_nature_evidence:
                raise ValueError("Simple clause labels do not require separate evidence spans")
        return
    previous = 0
    for clause in annotation.clauses:
        if not 0 <= clause.start < clause.end <= len(text) or clause.start < previous:
            raise ValueError("Clauses must be ordered, non-overlapping, valid character spans")
        if not text[clause.start:clause.end].strip():
            raise ValueError("A clause cannot contain only whitespace")
        if annotation.status == "reviewed" and any(c.isalnum() for c in text[previous:clause.start]):
            raise ValueError("Reviewed clauses must cover all words of the original utterance")
        previous = clause.end
        for label, evidence in [(clause.activity, clause.activity_evidence),
                                (clause.job_nature, clause.job_nature_evidence)]:
            if evidence and label is not True:
                raise ValueError("Evidence requires a positive label")
            for span in evidence:
                if not clause.start <= span.start < span.end <= clause.end:
                    raise ValueError("Evidence must lie inside its clause")
                if not text[span.start:span.end].strip():
                    raise ValueError("Evidence cannot be blank")
            if annotation.status == "reviewed" and (label is None or (label and not evidence)):
                raise ValueError("Reviewed clauses need both labels and evidence for each positive label")
    if annotation.status == "reviewed" and (not annotation.clauses or any(c.isalnum() for c in text[previous:])):
        raise ValueError("Reviewed annotations must cover the whole utterance")
