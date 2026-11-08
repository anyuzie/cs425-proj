"""Separate interactive clause inspector. Does not load the production pipeline."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from uuid import uuid4


def display(result):
    print("\nOriginal:       " + result["original_clause"])
    print("DiffuSeq:       " + (result["diffuseq_rewrite"] or "[no rewrite]"))
    print("Reconstruction: " + (result["reconstruction"] or "[not available]"))
    fidelity = result["fidelity"]
    if fidelity:
        print(f"Fidelity: direct={fidelity['direct_cosine']:.4f}, round-trip={fidelity['cycle_cosine']:.4f}")
        print("Rewrite: " + ("ACCEPTED" if result["rewrite_accepted"] else "REJECTED: " + ", ".join(fidelity["rejection_reasons"])))
        if result["rewrite_is_copy"]:
            print("Rewrite note: wording is unchanged apart from case/punctuation; not evidence of successful standardization.")
    for error in result["generation_errors"]:
        print("Rewrite unavailable: " + error)
    print("ColBERT input:  " + result["matching_text"])
    print("Input source:   " + result["matching_source"])
    print("Preference:     " + result["preference"])
    print("\nMatched occupation tasks (semantic associations, not suitability probabilities):")
    for task in result["task_matches"]:
        print(f"  {task['rank']}. {task['occupation_title']} [{task['occupation_code']}] | score={task['score']:.4f}")
        print(f"     Task {task['task_id']}: {task['task_text']}")
    if result["rejected_rewrite_task_preview"]:
        print("\nDiffuSeq rewrite task matches — PREVIEW ONLY; rejected rewrite was not used above:")
        for task in result["rejected_rewrite_task_preview"]:
            print(f"  {task['rank']}. {task['occupation_title']} | {task['score']:.4f} | {task['task_text']}")
    print(f"\nTime: {result['seconds']:.2f}s. Reconstruction is never a matching query.\n")


def main(argv=None):
    tool_root = Path(__file__).resolve().parents[2]
    local_path = tool_root / "runtime.local.json"
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    parser = argparse.ArgumentParser(description="Standalone DiffuSeq → fidelity checks → ColBERT task inspector")
    parser.add_argument("--clause", help="Process one clause and exit; omit for interactive input")
    parser.add_argument("--project-root", type=Path, default=local.get("project_root"))
    parser.add_argument("--resources-root", type=Path, default=local.get("resources_root", str(tool_root)))
    parser.add_argument("--encoder", type=Path, default=local.get("encoder"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000, choices=(50, 100, 200, 1000, 2000))
    parser.add_argument("--diffuseq-model", choices=("simplification", "qqp"), default="simplification")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--json", action="store_true", help="Print structured results instead of the human-readable view")
    args = parser.parse_args(argv)
    if args.project_root is None or args.encoder is None:
        parser.error("Configure runtime.local.json or supply --project-root and --encoder")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        from .generation import DiffuSeqGenerator, Reconstructor, FidelityEvaluator
        from .matching import ColbertTaskMatcher
        from .pipeline import ClausePipeline
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this interpreter. Use the separate lab launcher/environment.")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("Loading real DiffuSeq checkpoint...", file=sys.stderr, flush=True)
        generator = DiffuSeqGenerator(args.resources_root, args.device, steps=args.steps, checkpoint_domain=args.diffuseq_model)
        print("Loading separate FLAN-T5 reconstructor...", file=sys.stderr, flush=True)
        reconstructor = Reconstructor(args.resources_root / "resources/flan_t5_base", args.device)
        print("Loading frozen MPNet and NLI fidelity checks...", file=sys.stderr, flush=True)
        evaluator = FidelityEvaluator(args.encoder, args.project_root / "models/nli")
        print("Encoding occupation tasks with ColBERTv2 (in memory)...", file=sys.stderr, flush=True)
        matcher = ColbertTaskMatcher(args.resources_root / "resources/colbertv2", args.project_root, args.device)
        pipeline = ClausePipeline(generator, reconstructor, evaluator, matcher)
        report_dir = args.resources_root / "reports"
        report_dir.mkdir(exist_ok=True)
        log_path = report_dir / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8] + ".jsonl")
        metadata = {"event": "session", "generator": generator.metadata, "matcher": matcher.metadata,
                    "project_root": str(args.project_root), "encoder": str(args.encoder),
                    "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                    "dependencies": {name: version(name) for name in ("torch", "transformers", "sentence-transformers")},
                    "fidelity_thresholds": {"direct_cosine": .8, "cycle_cosine": .8, "bidirectional_entailment": .5},
                    "thresholds_calibrated": False}
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata) + "\n")
        print(f"Ready: {len(matcher.tasks)} tasks from {matcher.metadata['eligible_occupation_count']} occupations. Log: {log_path}", file=sys.stderr, flush=True)
        while True:
            try:
                text = args.clause if args.clause is not None else input("Clause (or /quit)> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip() == "/quit":
                break
            if not text.strip():
                if args.clause is not None:
                    raise ValueError("Clause cannot be empty")
                continue
            print("Generating three DiffuSeq candidates and checking fidelity; full sampling may take about a minute...", file=sys.stderr, flush=True)
            result = pipeline.process(text, args.top_k)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "result", **result}, ensure_ascii=False, allow_nan=False) + "\n")
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                display(result)
            if args.clause is not None:
                break
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Standalone lab failed: {error}", file=sys.stderr)
        return 1
