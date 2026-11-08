"""Independent-turn ranking with user-clause task/context diagnostics.

Run from the project root: .venv/Scripts/python.exe -m app.ranking_terminal
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.extractor import Extractor
from recommender.models.ranker import RankingPipeline
from recommender.schemas import TYPES
from recommender.utterance import UtteranceProcessingPipeline
from recommender.utterance.task_context import TaskContextClassifier
from recommender.utterance.stanza_pipeline import StanzaUtteranceProcessingPipeline
from app.utterance_collection import UtteranceCollection
from recommender.utterance.context_occupations import ContextOccupationMatcher
from recommender.utterance.context_matching import ContextMatchingPipeline, route_clause_pipelines, below_clause_score_floor
from recommender.utterance.pipeline import ProcessedUtterance


def rank_utterance(text, extractor, ranker, *, top_k=50):
    """Adapt one Reading to runtime inputs; no history or confidence merging."""
    reading = extractor.read(text)
    interests = dict.fromkeys(TYPES, 0.0)
    observed = set()
    for slot in reading.slots:
        interests[slot.slot] = slot.value
        observed.add(slot.slot)
    # This independent snapshot contains only the current user's utterance.
    evidence = [{"id": "e1", "turn": 1, "text": text, "status": "active"}]
    response = ranker.rank_with_evidence(
        interests, observed, [text], evidence, top_k=top_k)
    return reading, interests, observed, response


def show_result(reading, interests, observed, response):
    print("\nRIASEC: " + "  ".join(
        f"{t}={interests[t]:+.2f}" if t in observed else f"{t}=unknown"
        for t in TYPES))
    for slot in reading.slots:
        print(f"  {slot.slot}: confidence={slot.confidence:.4f}, reason={slot.reason}")
    if response.status == "no_signal":
        print(f"No ranking: {response.reason}\n")
        return
    displayed = response.occupations[:10]
    print(f"\nTop {len(displayed)} occupations:")
    for item in displayed:
        scores = []
        if item.rrf_score is not None:
            scores.append(f"RRF={item.rrf_score:.6f}")
        scores.extend(f"{name}={signal.score:.6f} (rank {signal.rank})"
                      for name, signal in item.signals.items())
        print(f"{item.rank:2}. {item.code} | {item.title}")
        print("    " + " | ".join(scores))

    print()


def show_clause_scores(processed, classifier, context_pipeline=None, context_occupations=None):
    """Score exact source clause text; display only, without changing ranking."""
    texts = [clause.text for clause in processed.clauses]
    predictions = classifier.predict(texts) if texts else []
    if len(predictions) != len(texts):
        raise ValueError("Task/context classifier returned a different number of scores than clauses")
    print("clauses: " + json.dumps(texts, ensure_ascii=False))
    for index, (clause, prediction) in enumerate(zip(processed.clauses, predictions), 1):
        print(f"  {index}. {clause.text}")
        print(f"     task_score={prediction['task_score']:.6f}"
              f" | context_score={prediction['context_score']:.6f}")
        print("     route_task=" + str(prediction['route_task']).lower()
              + " | route_context=" + str(prediction['route_context']).lower())

        if below_clause_score_floor(prediction):
            print("     skipped: both classifier scores are below 0.3")
        routed = route_clause_pipelines(clause.text, prediction, context_pipeline=context_pipeline)
        if "context" in routed:
            context = routed["context"]
            stance = context["preference"]
            sign = "+1" if stance["sign"] == 1 else "-1" if stance["sign"] == -1 else "unknown"
            print(f"     context preference: {stance['polarity']} ({sign}; {stance['rule']})")
            print("     context target: " + stance["target"])
            print("     context_vector (57 semantic similarities): " + json.dumps([round(v, 6) for v in context["context_vector"]]))
            print("     closest contexts: " + json.dumps(context["top_contexts"], ensure_ascii=False))
            if context_occupations is not None:
                matches = context_occupations.match(context)
                interpretation = ("potential preference conflicts" if stance["sign"] == -1 else
                                  "preference unresolved; associations only" if stance["sign"] is None else
                                  "context associations; not preference compatibility")
                print(f"     Context-associated occupations ({interpretation}):")
                if not matches:
                    print("       No usable context ratings for the selected attributes.")
                for rank, match in enumerate(matches, 1):
                    print(f"       {rank}. {match['code']} | {match['title']} | association={match['association_score']:.6f} | coverage={match['coverage']:.1%}")
                    for evidence in match["evidence"]:
                        print(f"          {evidence['context']}: rating={evidence['rating']:.3f}, similarity={evidence['similarity']:.3f}")
    print()
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-config', type=Path, help='Terminal preset; explicit CLI arguments take precedence')
    parser.add_argument('--text', help='Rank one independent utterance and exit (empty text reports no signal)')
    parser.add_argument('--json', action='store_true', help='Machine-readable one-shot response; requires --text')
    parser.add_argument('--top-k', type=int, default=50, help='One-shot ranking depth')
    parser.add_argument('--clause-diagnostics', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable the independent legacy BERT/MPNet clause diagnostics')
    parser.add_argument("--clause-splitter", choices=("stanza", "spacy"), default=None,
                        help="User-clause splitter; Stanza conservative baseline by default")
    parser.add_argument("--system-config", type=Path, default=None)
    parser.add_argument("--ranking-config", type=Path, help="Override the system's ranking configuration")
    parser.add_argument("--checkpoint", type=Path, help="Override the extractor checkpoint directory")
    parser.add_argument("--occupations", type=Path, default=ROOT / "data/occupations.csv")
    parser.add_argument("--task-context-classifier", type=Path, default=ROOT / "models/task_context_bert",
                        help="Trained independent task/context sigmoid classifier for user clauses")
    parser.add_argument("--context-dimensions", type=Path,
                        default=ROOT / "data/work_context/onet_31_0/compact/context_dimensions.csv")
    parser.add_argument("--collection-dir", type=Path, default=None,
                        help="Directory for append-only utterance snapshots")
    args = parser.parse_args()
    if args.json and args.text is None:
        parser.error('--json requires --text')
    # Existing ranking variants resolve model/cache paths relative to the repo.
    os.chdir(ROOT)
    try:
        terminal_config = json.loads(args.terminal_config.read_text(encoding='utf-8')) if args.terminal_config else {}
        if (not isinstance(terminal_config, dict) or set(terminal_config) - {'system_config','clause_diagnostics','clause_splitter','collection_dir'}
                or not isinstance(terminal_config.get('clause_diagnostics', True), bool)
                or terminal_config.get('clause_splitter', 'stanza') not in {'stanza','spacy'}):
            raise ValueError('invalid terminal configuration')
        args.system_config = args.system_config or Path(terminal_config.get('system_config', ROOT/'configs/system.json'))
        args.collection_dir = args.collection_dir or Path(terminal_config.get('collection_dir', ROOT/'data/collected/utterances'))
        args.clause_splitter = args.clause_splitter or terminal_config.get('clause_splitter', 'stanza')
        diagnostics = args.clause_diagnostics if args.clause_diagnostics is not None else terminal_config.get('clause_diagnostics', True)
        config = json.loads(args.system_config.read_text(encoding="utf-8"))
        if config["components"]["extractor"] != "deberta":
            raise ValueError("This script requires the configured deberta extractor")
        ranking_config = args.ranking_config or Path(config["paths"]["ranking_config"])
        checkpoint = args.checkpoint or Path(config["paths"]["extractor_checkpoint"])
        print(f"Loading ranking configuration: {ranking_config}", flush=True, file=sys.stderr if args.json else sys.stdout)
        ranker = RankingPipeline(args.occupations, config=ranking_config)
        print(f"Loading extractor: {checkpoint}", flush=True, file=sys.stderr if args.json else sys.stdout)
        extractor = Extractor(checkpoint, ROOT / "data/labels.json")
        if args.text is not None:
            from dataclasses import asdict
            reading, interests, observed, response = rank_utterance(args.text, extractor, ranker, top_k=args.top_k)
            if args.json:
                print(json.dumps({'text': args.text, 'reading': asdict(reading), 'interests': interests,
                                  'observed_types': sorted(observed), 'response': asdict(response)}, ensure_ascii=False))
            else:
                show_result(reading, interests, observed, response)
                print(json.dumps(response.query_diagnostics, indent=2, ensure_ascii=False))
            return 0
        utterance_processor = task_context_classifier = context_pipeline = context_occupations = None
        if diagnostics:
            print(f"Loading user-clause splitter: {args.clause_splitter}", flush=True)
            utterance_processor = (StanzaUtteranceProcessingPipeline() if args.clause_splitter == "stanza"
                                   else UtteranceProcessingPipeline())
            print(f"Loading user-clause classifier: {args.task_context_classifier}", flush=True)
            task_context_classifier = TaskContextClassifier(args.task_context_classifier)
            context_pipeline = ContextMatchingPipeline(args.context_dimensions)
            context_occupations = ContextOccupationMatcher(
                args.context_dimensions.parent / "occupation_context_vectors.csv",
                args.context_dimensions.parent / "occupation_context_mask.csv", args.occupations,
                context_pipeline.metadata["dimension_ids"])
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1
    try:
        collection = UtteranceCollection(args.collection_dir, {
            "ranking_config": ranker.resolved_config,
            "extractor_checkpoint": str(checkpoint.resolve()),
            'terminal_config': terminal_config,
            'clause_diagnostics': diagnostics,
            "context_pipeline": context_pipeline.metadata if diagnostics else None,
            "context_occupation_matching": context_occupations.metadata if diagnostics else None,
            "clause_score_floor": 0.3,
            "task_context_classifier_checkpoint": str(args.task_context_classifier.resolve()),
            "task_context_thresholds": {"task": task_context_classifier.task_threshold,
                                        "context": task_context_classifier.context_threshold} if diagnostics else None,
            "parser_model": utterance_processor.model if diagnostics else None,
            "parser_version": (version("stanza") if args.clause_splitter == "stanza" else utterance_processor.nlp.meta.get("version")) if diagnostics else None,
            "clause_splitter": args.clause_splitter,
            "spacy_version": version("spacy") if diagnostics else None,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("recommender/utterance/pipeline.py", "recommender/utterance/classification.py",
                             "app/ranking_terminal.py", "app/utterance_collection.py",
                             "recommender/utterance/task_context.py", "recommender/utterance/stanza_pipeline.py", "recommender/utterance/context_matching.py", "recommender/utterance/context_occupations.py")},
        })
    except (OSError, ValueError, TypeError) as error:
        print(f"Unable to start data collection: {error}", file=sys.stderr)
        return 1
    print(f"Collecting utterance snapshots: {collection.path}")
    enabled = [f"{k}={v}" for k, v in ranker.component_identifiers.items() if v]
    print("\nEnabled: " + ", ".join(enabled))
    print(f"Eligible occupations: {ranker.eligible_count}")
    print("Each entry is independent. Scores are raw component scores, not probabilities.")
    if diagnostics:
        print("Clause scores are independent BERT sigmoids; routing uses checkpoint thresholds.")
        print("Experimental clause classification is display-only and does not affect ranking.")
    else:
        print('Legacy clause diagnostics disabled; they never affect ranking.')
    print("Enter an utterance, or x to exit.")
    while True:
        try:
            raw_text = input("You> ")
            text = raw_text.strip()
            if text == "x":
                break
            if not text:
                continue
            try:
                entry_id = collection.submitted(raw_text, text)
            except OSError as error:
                print(f"Unable to save utterance; stopping collection: {error}", file=sys.stderr)
                return 1
            processed = None
            try:
                processed = utterance_processor.split(text) if diagnostics else ProcessedUtterance(text, ())
                result = rank_utterance(text, extractor, ranker)
                collection.completed(entry_id, processed, *result)
            except (Exception, KeyboardInterrupt) as error:
                try:
                    collection.failed(entry_id, error, processed)
                except OSError as save_error:
                    print(f"Unable to save processing failure: {save_error}", file=sys.stderr)
                    return 1
                raise
            show_result(*result)
            if diagnostics:
                show_clause_scores(processed, task_context_classifier, context_pipeline, context_occupations)
        except (EOFError, KeyboardInterrupt):
            break
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Unable to rank this entry: {error}", file=sys.stderr)
    print("Goodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
