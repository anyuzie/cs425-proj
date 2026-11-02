"""Context-associated occupation evidence, independent of RIASEC ranking."""
import csv
import hashlib
from pathlib import Path
import math


class ContextOccupationMatcher:
    def __init__(self, vectors_path, mask_path, occupations_path, dimension_ids):
        self.dimension_ids = list(dimension_ids)
        self.metadata = {"variant": "context_association_top3_v1", "dimensions": self.dimension_ids,
                         "scope": "all_local_occupations", "aggregation": "available_similarity_weighted_rating"}
        def read(path):
            path = Path(path)
            self.metadata[str(path.name) + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            return fields, rows
        _, occupations = read(occupations_path)
        self.titles = {r["code"]: r["Title"] for r in occupations}
        fields, vectors = read(vectors_path)
        mask_fields, masks = read(mask_path)
        if fields != ["occupation_code"] + self.dimension_ids or mask_fields != fields:
            raise ValueError("Context vectors and masks must match dimension order exactly")
        vector_map = {r["occupation_code"]: r for r in vectors}
        mask_map = {r["occupation_code"]: r for r in masks}
        if (len(vector_map) != len(vectors) or len(mask_map) != len(masks)
                or set(vector_map) != set(mask_map) or set(vector_map) != set(self.titles)):
            raise ValueError("Occupation coverage/uniqueness differs between context sources")
        self.ratings = {}
        for code, row in vector_map.items():
            values = {}
            for key in self.dimension_ids:
                flag = mask_map[code][key]
                if flag not in {"0", "1"}:
                    raise ValueError("Invalid context availability mask")
                if flag == "0":
                    if row[key] != "":
                        raise ValueError("Unavailable context rating must be blank")
                    continue
                value = float(row[key])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Context rating must be finite and normalized")
                values[key] = value
            self.ratings[code] = values

    def match(self, context, limit=10):
        if type(limit) is not int or limit < 1:
            raise ValueError("Display limit must be positive")
        attributes = [a for a in context["top_contexts"] if a["similarity"] > 0]
        if any(a["context_id"] not in self.dimension_ids for a in attributes):
            raise ValueError("Unknown context dimension")
        total_weight = sum(a["similarity"] for a in attributes)
        matches = []
        for code, ratings in self.ratings.items():
            evidence = [{**a, "rating": ratings[a["context_id"]]}
                        for a in attributes if a["context_id"] in ratings]
            weight = sum(a["similarity"] for a in evidence)
            if not weight:
                continue
            score = sum(a["similarity"] * a["rating"] for a in evidence) / weight
            matches.append({"code": code, "title": self.titles[code], "association_score": score,
                            "coverage": weight / total_weight, "evidence": evidence})
        matches.sort(key=lambda r: (-r["association_score"], -r["coverage"], r["code"]))
        return matches[:limit]
