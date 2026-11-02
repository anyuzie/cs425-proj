"""Fine-tune BERT/RoBERTa using reviewed JSONL clauses and independent BCE outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random

LABELS = ("activity", "job_nature")


def load_examples(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("training_eligible") is False:
            continue
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Every training example needs nonempty text")
        labels = row.get("labels", {})
        if set(labels) != set(LABELS) or any(labels[k] not in (0, 1, False, True) for k in LABELS):
            raise ValueError("Every example needs reviewed activity and job_nature binary labels")
        rows.append({"text": row["text"], "labels": [float(labels[k]) for k in LABELS],
                     "group": row.get("occupation_code", row.get("group_id"))})
    if not rows:
        raise ValueError("No eligible labelled examples")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, validation = load_examples(args.train), load_examples(args.validation)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting a checkpoint")
    normalize = lambda text: " ".join(text.lower().split())
    if {normalize(r["text"]) for r in train} & {normalize(r["text"]) for r in validation}:
        raise ValueError("Training and validation contain duplicate clauses")
    if {r["group"] for r in train if r["group"]} & {r["group"] for r in validation if r["group"]}:
        raise ValueError("Keep occupation/session groups in separate splits")
    if any({r["labels"][i] for r in train} != {0.0, 1.0} for i in range(2)):
        raise ValueError("Training needs positive and negative examples for both labels")
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, set_seed
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.base_model)
    if config.model_type not in {"bert", "roberta"}:
        raise ValueError("Choose a BERT or RoBERTa base checkpoint")
    config.num_labels = 2
    config.id2label = dict(enumerate(LABELS))
    config.label2id = {label: i for i, label in enumerate(LABELS)}
    config.problem_type = "multi_label_classification"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for start in range(0, len(train), args.batch_size):
            rows = train[start:start + args.batch_size]
            encoded = tokenizer([r["text"] for r in rows], padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(device)
            labels = torch.tensor([r["labels"] for r in rows], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"Completed epoch {epoch + 1}", flush=True)
    model.eval()
    tp, fp, fn = [0, 0], [0, 0], [0, 0]
    exact = 0
    with torch.inference_mode():
        for start in range(0, len(validation), args.batch_size):
            rows = validation[start:start + args.batch_size]
            encoded = tokenizer([r["text"] for r in rows], padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(device)
            predictions = model(**encoded).logits.sigmoid().ge(0.5).cpu().tolist()
            for prediction, row in zip(predictions, rows):
                exact += prediction == row["labels"]
                for i in range(2):
                    tp[i] += bool(prediction[i] and row["labels"][i])
                    fp[i] += bool(prediction[i] and not row["labels"][i])
                    fn[i] += bool(not prediction[i] and row["labels"][i])
    metrics = {label: {"f1": 2*tp[i]/max(1, 2*tp[i]+fp[i]+fn[i]),
                        "tp": tp[i], "fp": fp[i], "fn": fn[i]} for i, label in enumerate(LABELS)}
    metrics["exact_match"] = exact / len(validation)
    model.config.occupation_classifier = {"trained": True, "thresholds": [0.5, 0.5], "max_length": 256}
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    manifest = {"base_model": args.base_model, "seed": args.seed, "epochs": args.epochs,
                "batch_size": args.batch_size, "validation": metrics,
                "data_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in (("train", args.train), ("validation", args.validation))}}
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
