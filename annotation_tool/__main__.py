"""Run only the local annotation tool or export reviewed training examples."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from .store import Store
from .export import export_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve", "export", "serve-occupations"], nargs="?", default="serve")
    parser.add_argument("--collection-dir", type=Path, default=ROOT / "data/collected/utterances")
    parser.add_argument("--annotation-dir", type=Path, default=ROOT / "data/annotations/utterances")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/utterances")
    parser.add_argument("--port", type=int)
    parser.add_argument("--occupation-csv", type=Path, help="Authoritative occupation review CSV")
    parser.add_argument("--occupation-batch", type=Path, help="CSV containing first-batch clause IDs (read only)")
    args = parser.parse_args()
    if args.command == "serve-occupations":
        if args.occupation_csv is None:
            parser.error("serve-occupations requires --occupation-csv")
        import uvicorn
        from .occupation_review import create_occupation_app
        batch = args.occupation_batch or args.occupation_csv.with_name("review_first_100.csv")
        port = args.port if args.port is not None else 8011
        print(f"Occupation review: http://127.0.0.1:{port}")
        uvicorn.run(create_occupation_app(args.occupation_csv, batch), host="127.0.0.1", port=port)
        return
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        port = args.port if args.port is not None else 8010
        print(f"Annotation tool: http://127.0.0.1:{port}")
        uvicorn.run(create_app(args.collection_dir, args.annotation_dir, args.output), host="127.0.0.1", port=port)
    else:
        rows, manifest = export_rows(Store(args.collection_dir, args.annotation_dir))
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        manifest.update(created_at=datetime.now(timezone.utc).isoformat(),
                        dataset_sha256=sha256(payload.encode("utf-8")).hexdigest())
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".jsonl")
        with destination.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Exported {len(rows)} reviewed clauses to {destination}")


if __name__ == "__main__":
    main()
