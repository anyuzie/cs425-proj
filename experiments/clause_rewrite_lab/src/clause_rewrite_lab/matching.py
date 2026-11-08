"""Frozen ColBERTv2 token encodings and exact MaxSim over raw task text."""
import csv
from pathlib import Path
import string


def windows(tokens, content_limit, overlap):
    if not tokens:
        return [[]]
    result = []
    for start in range(0, len(tokens), content_limit - overlap):
        result.append(tokens[start:start + content_limit])
        if start + content_limit >= len(tokens):
            break
    return result


def maxsim(query, documents, mask):
    """Same masked document-max, query-sum reduction as upstream ColBERT."""
    scores = documents @ query.T
    scores = scores.masked_fill(~mask.unsqueeze(-1), -9999.)
    return scores.max(dim=1).values.sum(dim=-1)


class ColbertTaskMatcher:
    def __init__(self, model_path, project_root, device="cuda", batch_size=24):
        import torch
        from transformers import AutoTokenizer, BertConfig, BertModel
        from safetensors.torch import load_file
        self.device, self.batch_size = device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        config = BertConfig.from_pretrained(model_path, local_files_only=True)
        config._attn_implementation = "eager"
        self.bert = BertModel(config)
        weights = load_file(str(Path(model_path) / "model.safetensors"))
        self.projection = torch.nn.Linear(config.hidden_size, weights["linear.weight"].shape[0], bias=False)
        self.projection.load_state_dict({"weight": weights["linear.weight"]}, strict=True)
        bert_weights = {name.removeprefix("bert."): value for name, value in weights.items() if name.startswith("bert.")}
        # Old checkpoints persisted position_ids, which modern BertModel derives.
        bert_weights.pop("embeddings.position_ids", None)
        self.bert.load_state_dict(bert_weights, strict=True)
        self.bert.eval().requires_grad_(False).to(device)
        self.projection.eval().requires_grad_(False).to(device)
        self.query_marker = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.doc_marker = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.punctuation = {self.tokenizer.encode(char, add_special_tokens=False)[0] for char in string.punctuation}
        self.tasks = self._load_tasks(Path(project_root))
        self.document_vectors, self.document_task_ids = [], []
        self._prepare_tasks()
        self.metadata = {"model": "colbert-ir/colbertv2.0", "dimension": self.projection.out_features,
                         "eligible_occupation_count": len({task['occupation_code'] for task in self.tasks}),
                         "task_count": len(self.tasks), "document_windows": len(self.document_vectors),
                         "score": "mean query-window MaxSim / 32; best document window per task",
                         "query_maxlen": 32, "doc_maxlen": 180, "disk_task_embeddings": False}

    @staticmethod
    def _load_tasks(project):
        import math
        with (project / "data/occupations.csv").open(encoding="utf-8-sig", newline="") as handle:
            occupations = {row["code"]: row["Title"] for row in csv.DictReader(handle)
                           if row.get("job_zone") and float(row["job_zone"]) in (4., 5.)
                           and all(row.get("z_" + key) and math.isfinite(float(row["z_" + key])) for key in "RIASEC")}
        tasks, seen = [], set()
        with (project / "data/tasks.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["code"] not in occupations:
                    continue
                key = (row["code"], row["Task ID"])
                if key in seen or not row["Task"].strip():
                    raise ValueError("Duplicate or empty task")
                seen.add(key)
                tasks.append({"occupation_code": row["code"], "occupation_title": occupations[row["code"]],
                              "task_id": row["Task ID"], "task_text": row["Task"].strip()})
        return sorted(tasks, key=lambda task: (task["occupation_code"], task["task_id"]))

    def _encode(self, token_groups, is_query=False):
        import torch
        tokenizer = self.tokenizer
        rows, attention, content_masks = [], [], []
        length = 32 if is_query else max(len(group) + 3 for group in token_groups)
        for group in token_groups:
            ids = [tokenizer.cls_token_id, self.query_marker if is_query else self.doc_marker, *group, tokenizer.sep_token_id]
            real = len(ids)
            ids += [tokenizer.mask_token_id if is_query else tokenizer.pad_token_id] * (length - real)
            # Preserve ColBERT's checkpoint default: MASK augmentation embeddings
            # participate in MaxSim, but padding positions do not attend as keys.
            rows.append(ids)
            attention.append([1] * real + [0] * (length - real))
            content_masks.append([token != tokenizer.pad_token_id and token not in self.punctuation for token in ids])
        ids = torch.tensor(rows, device=self.device)
        with torch.inference_mode():
            hidden = self.bert(ids, attention_mask=torch.tensor(attention, device=self.device)).last_hidden_state
            vectors = torch.nn.functional.normalize(self.projection(hidden), dim=-1)
        mask = torch.tensor(content_masks, device=self.device)
        if not is_query:
            vectors = vectors * mask.unsqueeze(-1)
        return vectors, mask

    def _prepare_tasks(self):
        groups, task_indices = [], []
        for index, task in enumerate(self.tasks):
            tokens = self.tokenizer.encode(task["task_text"], add_special_tokens=False)
            for group in windows(tokens, 177, 32):
                groups.append(group)
                task_indices.append(index)
        for start in range(0, len(groups), self.batch_size):
            vectors, masks = self._encode(groups[start:start + self.batch_size])
            for values, mask in zip(vectors, masks):
                self.document_vectors.append(values[mask].cpu())
        self.document_task_ids = task_indices

    def match(self, query_text, top_k=5):
        import torch
        if not query_text.strip():
            return []
        groups = windows(self.tokenizer.encode(query_text, add_special_tokens=False), 29, 8)
        query_vectors, _ = self._encode(groups, is_query=True)
        all_scores = torch.zeros(len(self.document_vectors), device=self.device)
        for query in query_vectors:
            for start in range(0, len(self.document_vectors), self.batch_size):
                batch = self.document_vectors[start:start + self.batch_size]
                lengths = torch.tensor([len(doc) for doc in batch], device=self.device)
                docs = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True).to(self.device)
                mask = torch.arange(docs.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
                all_scores[start:start + len(batch)] += maxsim(query, docs, mask) / (32 * len(groups))
        task_scores = {}
        for index, score in zip(self.document_task_ids, all_scores.cpu().tolist()):
            task_scores[index] = max(score, task_scores.get(index, float("-inf")))
        ordered = sorted(task_scores, key=lambda index: (-task_scores[index], self.tasks[index]["occupation_code"], self.tasks[index]["task_id"]))
        return [{**self.tasks[index], "score": task_scores[index], "rank": rank}
                for rank, index in enumerate(ordered[:top_k], 1)]
