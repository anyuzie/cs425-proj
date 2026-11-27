"""Train a three-output NLI checkpoint or compare stock and tuned checkpoints.

Run python scripts/finetune_nli.py --help. No training or downloads on import.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.nli import BINARY, metrics, select_threshold
from pipeline.nli import MODEL_NAME, NliConfirmer


def read_data(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids, pairs, groups = set(), {}, {}
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Every row needs a unique string id")
        ids.add(row["id"])
        if row.get("split") not in {"train", "dev", "test"} or row.get("label") not in BINARY:
            raise ValueError(f"Invalid split or label: {row['id']}")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("a", "b")):
            raise ValueError(f"Missing sentence: {row['id']}")
        pair = tuple(sorted(" ".join(row[k].casefold().split()) for k in ("a", "b")))
        if pair in pairs and pairs[pair] != row["split"]:
            raise ValueError(f"Pair or reversed pair crosses splits: {row['id']}")
        pairs[pair] = row["split"]
        group = row.get("group_id")
        if group and group in groups and groups[group] != row["split"]:
            raise ValueError(f"Related group crosses splits: {group}")
        if group:
            groups[group] = row["split"]
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if r["split"] == split]
        if not subset or {BINARY[r["label"]] for r in subset} != {0, 1}:
            raise ValueError(f"{split} requires both binary classes")
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def binary_metrics(labels, scores, threshold):
    result = metrics(labels, scores, threshold)
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]
    return {**result, "accuracy": (tn + tp) / len(labels)}


def binary_loss(logits, targets, contradiction_index):
    import torch
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    others = [i for i in range(3) if i != contradiction_index]
    return torch.where(targets.bool(), -log_probs[:, contradiction_index],
                       -torch.logsumexp(log_probs[:, others], dim=-1)).mean()


def provenance(args):
    import importlib.metadata
    return {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")}}


def train(args, rows):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if any(r.get("reviewed") is False for r in rows if r["split"] == "train") and not args.allow_unreviewed:
        raise ValueError("Unreviewed synthetic training rows remain. Review them or explicitly pass --allow-unreviewed for a pilot.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(device)
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if set(labels) != {0, 1, 2} or set(labels.values()) != {"contradiction", "entailment", "neutral"}:
        raise ValueError(f"Unverified label mapping: {labels}")
    ci = next(i for i in labels if labels[i] == "contradiction")

    def collate(batch):
        inputs = tokenizer([r["a"] for r in batch], [r["b"] for r in batch],
                           padding=True, truncation=True, max_length=512, return_tensors="pt")
        return inputs.to(device), torch.tensor([BINARY[r["label"]] for r in batch], device=device)

    loaders = {split: DataLoader([r for r in rows if r["split"] == split],
                                batch_size=args.batch_size, shuffle=split == "train", collate_fn=collate)
               for split in ("train", "dev")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=False)
    # Store exact inputs and selection policy before optimisation. Test is never inferred here.
    (args.output / "dataset.jsonl").write_bytes(args.data.read_bytes())
    manifest = {**provenance(args), "device": str(device), "label_order": labels,
                "base_commit": getattr(model.config, "_commit_hash", None),
                "selection": "lowest dev binary negative log likelihood; earlier epoch wins ties",
                "objective": "contradiction vs summed entailment/neutral probability", "epochs": []}
    write_json(args.output / "training.json", manifest)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, count = 0., 0
            with torch.set_grad_enabled(split == "train"):
                for batch_index, (inputs, targets) in enumerate(loader, 1):
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                    loss = binary_loss(model(**inputs).logits, targets, ci)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training/development loss")
                    if split == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                    total += loss.item() * len(targets)
                    count += len(targets)
                    if batch_index % 10 == 0 or batch_index == len(loader):
                        print(f"Epoch {epoch} {split}: {count}/{len(loader.dataset)} pairs; loss {total/count:.4f}", flush=True)
            losses[split] = total / count
        if losses["dev"] < best:
            best = losses["dev"]
            model.save_pretrained(args.output / "checkpoint")
            tokenizer.save_pretrained(args.output / "checkpoint")
            manifest["best_epoch"] = epoch
        manifest["epochs"].append({"epoch": epoch, **losses})
        write_json(args.output / "training.json", manifest)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}; dev loss {losses['dev']:.4f}", flush=True)
    print(f"Best checkpoint: {args.output / 'checkpoint'}")


def compare(args, rows):
    import gc
    args.output.mkdir(parents=True, exist_ok=False)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    scores, winners, thresholds = {}, {}, {}
    info = provenance(args)
    tuned_manifest = Path(args.tuned).parent / "training.json"
    if tuned_manifest.exists():
        training = json.loads(tuned_manifest.read_text(encoding="utf-8"))
        if training["data_sha256"] != info["data_sha256"]:
            raise ValueError("Dataset changed since training; use the saved training dataset snapshot or retrain.")
        info["training"] = training
    # Select both thresholds and persist them before any test inference.
    for name, path in (("stock", args.base), ("tuned", args.tuned)):
        confirmer = NliConfirmer(str(path))
        scores[name] = {}
        winners[name] = {}
        for row in dev:
            probs = confirmer.probabilities(row["a"], row["b"])
            scores[name][row["id"]] = probs
            winners[name][row["id"]] = max(probs, key=probs.get)
        threshold, _ = select_threshold([BINARY[r["label"]] for r in dev],
                                       [scores[name][r["id"]]["contradiction"] for r in dev])
        thresholds[name] = threshold
        del confirmer
        gc.collect()
    write_json(args.output / "thresholds.json", {**info, "thresholds": thresholds,
               "comparison": "P(contradiction) > threshold", "selection": "dev F1; ties precision then higher threshold"})
    results = {}
    for name, path in (("stock", args.base), ("tuned", args.tuned)):
        confirmer = NliConfirmer(str(path))
        for row in test:
            probs = confirmer.probabilities(row["a"], row["b"])
            scores[name][row["id"]] = probs
            winners[name][row["id"]] = max(probs, key=probs.get)
        results[name] = {}
        for split, subset in (("dev", dev), ("test", test)):
            ys = [BINARY[r["label"]] for r in subset]
            ps = [scores[name][r["id"]]["contradiction"] for r in subset]
            results[name][split] = {
                "threshold_0.60": binary_metrics(ys, ps, .6),
                "dev_selected_threshold": binary_metrics(ys, ps, thresholds[name]),
                "argmax_binary_accuracy": sum((winners[name][r["id"]] == "contradiction") == bool(y)
                                               for r, y in zip(subset, ys)) / len(ys)}
        del confirmer
        gc.collect()
    exported = []
    for row in dev + test:
        record = {k: row.get(k, "") for k in ("id", "split", "source", "slot", "label", "a", "b")}
        y = BINARY[row["label"]]
        for name in ("stock", "tuned"):
            record.update({f"{name}_{label}": value for label, value in scores[name][row["id"]].items()})
            record[f"{name}_class"] = winners[name][row["id"]]
            record[f"{name}_correct"] = int((scores[name][row["id"]]["contradiction"] > thresholds[name]) == bool(y))
        record["change"] = ("fixed" if record["tuned_correct"] else "regressed") if record["stock_correct"] != record["tuned_correct"] else "unchanged"
        exported.append(record)
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    summary = {**info, "thresholds": thresholds, "metrics": results,
               "test_changes": {kind: sum(r["change"] == kind and r["split"] == "test" for r in exported)
                                for kind in ("fixed", "regressed", "unchanged")}}
    write_json(args.output / "comparison.json", summary)
    print(json.dumps({"thresholds": thresholds, "test": {n: r["test"] for n, r in results.items()},
                      "test_changes": summary["test_changes"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        fromfile_prefix_chars="@",
        epilog="Use @PATH to read saved arguments (one argument per line).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "compare"):
        p = sub.add_parser(command)
        p.add_argument("--data", type=Path, default=ROOT / "data/nli_pairs.jsonl")
        p.add_argument("--base", default=MODEL_NAME)
        p.add_argument("--output", type=Path, required=True, help="New directory; existing directories are never overwritten")
        if command == "train":
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--learning-rate", type=float, default=1e-5)
            p.add_argument("--seed", type=int, default=425)
            p.add_argument("--allow-unreviewed", action="store_true")
        else:
            p.add_argument("--tuned", type=Path, required=True, help="Saved checkpoint directory")
    args = parser.parse_args()
    if args.command == "train" and (args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1):
        parser.error("epochs/batch size must be positive; learning rate must be between 0 and 1")
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    try:
        rows = read_data(args.data)
        (train if args.command == "train" else compare)(args, rows)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
