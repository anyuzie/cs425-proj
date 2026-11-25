"""Occupation content annotation, independent of ranking and user preference."""
import csv
import math
from pathlib import Path

LABELS = ("activity", "job_nature")


class OccupationClauseClassifier:
    def __init__(self, checkpoint, batch_size=32):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        checkpoint = Path(checkpoint)
        if not (checkpoint / "config.json").is_file():
            raise ValueError(f"Occupation classifier checkpoint missing: {checkpoint}. "
                             "Train it with python -m training.occupation_classifier first.")
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if (config.model_type not in {"bert", "roberta"} or config.num_labels != 2
                or config.problem_type != "multi_label_classification"
                or tuple(config.id2label.get(i) for i in range(2)) != LABELS):
            raise ValueError("Expected a BERT/RoBERTa multi-label checkpoint with activity/job_nature outputs")
        settings = getattr(config, "occupation_classifier", {})
        if settings.get("trained") is not True:
            raise ValueError("Occupation checkpoint must be trained; a random head is not usable")
        self.thresholds = settings.get("thresholds", [0.5, 0.5])
        if len(self.thresholds) != 2 or any(not math.isfinite(x) or not 0 < x < 1 for x in self.thresholds):
            raise ValueError("Expected two finite thresholds between zero and one")
        self.max_length = settings.get("max_length", 256)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True).to(self.device).eval()

    def predict(self, texts):
        import torch
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                   truncation=True, max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                probabilities = self.model(**batch.to(self.device)).logits.sigmoid().cpu().tolist()
            for scores in probabilities:
                if len(scores) != 2 or any(not math.isfinite(x) for x in scores):
                    raise ValueError("Invalid occupation classifier probabilities")
                results.append({"probabilities": dict(zip(LABELS, scores)),
                                "labels": {label: score >= threshold for label, score, threshold
                                           in zip(LABELS, scores, self.thresholds)}})
        return results


class OccupationAnnotations:
    """Classify descriptions/tasks lazily; cache only within this model/data instance."""
    def __init__(self, occupations, tasks, splitter, classifier):
        self.splitter, self.classifier = splitter, classifier
        self.sources, self.cache = {}, {}
        with Path(occupations).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                self.sources[row["code"]] = [("Description", row.get("Description", ""))]
        if tasks is not None:
            with Path(tasks).open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["code"] in self.sources:
                        self.sources[row["code"]].append(("Task:" + row["Task ID"], row["Task"]))

    def get(self, code):
        if code not in self.cache:
            clauses, seen = [], set()
            for field, text in self.sources.get(code, []):
                # Occupation prose keeps coordinated actions/shared arguments
                # together. User-utterance splitting is separate.
                if getattr(self.splitter, 'nlp', None) is not None:
                    spans = []
                    for sentence in self.splitter.nlp(text).sents:
                        value = sentence.text.strip()
                        if value:
                            start = sentence.start_char + len(sentence.text) - len(sentence.text.lstrip())
                            spans.append((value, start, start + len(value)))
                else:
                    spans = [(c.text, c.start, c.end) for c in self.splitter.split(text).clauses]
                for value, start, end in spans:
                    key = (field, start, end)
                    if key not in seen:
                        seen.add(key)
                        clauses.append({"text": value, "source_field": field,
                                        "start": start, "end": end})
            predictions = self.classifier.predict([c["text"] for c in clauses])
            if len(predictions) != len(clauses):
                raise ValueError("Occupation classifier returned a different number of predictions")
            self.cache[code] = [dict(clause, **prediction) for clause, prediction in zip(clauses, predictions)]
        return self.cache[code]
