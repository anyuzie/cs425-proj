"""Explicit M4 model download and immutable task-cache build; never invoked by a turn."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from time import perf_counter

from .data_loader import load_eligible_occupations
from .dense_config import BASELINE_PARAMETERS


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_model(parameters):
    from huggingface_hub import snapshot_download
    from .dense_ranker import digest, model_identity
    directory = Path(parameters["model_dir"])
    if (directory / "model_source.json").exists():
        model_identity(parameters)
        return
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(parameters["model_id"], revision=parameters["model_revision"], local_dir=directory,
                      allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*.json"],
                      max_workers=2)
    files = {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and ".cache" not in p.parts and p.name != "model_source.json"}
    if not (directory / "model.safetensors").is_file():
        raise ValueError("downloaded model snapshot has no safetensors weights")
    save(directory / "model_source.json", {"model_id": parameters["model_id"],
                                           "revision": parameters["model_revision"], "files": files})


def build(occupations, parameters, *, encoder=None, model_hash=None):
    import faiss
    import numpy as np
    from .dense_ranker import MiniLMEncoder, cache_identity, digest, fingerprint, load_tasks, model_identity, validated_vectors
    if parameters != BASELINE_PARAMETERS:
        raise ValueError("build requires the frozen minilm@mean_tasks_v1 parameters")
    directory = Path(parameters["cache_dir"])
    if directory.exists():
        raise ValueError(f"cache already exists: {directory}; preserve it and use a new variant for changes")
    started = perf_counter()
    tasks = load_tasks(parameters["tasks_path"], {o.code for o in occupations})
    identity = cache_identity(occupations, parameters, tasks, model_hash or model_identity(parameters))
    encoder = encoder or MiniLMEncoder(parameters)
    encode_started = perf_counter()
    vectors = validated_vectors(encoder.encode([task.text for task in tasks]), len(tasks), parameters["dimension"])
    encode_seconds = perf_counter() - encode_started
    index_started = perf_counter()
    index = faiss.IndexFlatIP(parameters["dimension"])
    index.add(vectors)
    index_seconds = perf_counter() - index_started
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".m4-build-{uuid.uuid4().hex}"
    temporary.mkdir()  # Inherit project permissions; tempfile directories may be owner-only on Windows.
    try:
        np.save(temporary / "embeddings.npy", vectors, allow_pickle=False)
        faiss.write_index(index, str(temporary / "index.faiss"))
        save(temporary / "owners.json", [asdict(task) for task in tasks])
        artifacts = {name: digest(temporary / name) for name in ("embeddings.npy", "index.faiss", "owners.json")}
        metadata = {"identity": identity, "identity_sha256": fingerprint(identity), "artifact_sha256": artifacts,
                    "task_count": len(tasks), "occupation_count": len({t.code for t in tasks}),
                    "missing_task_occupations": sorted({o.code for o in occupations} - {t.code for t in tasks}),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "encode_seconds": encode_seconds,
                    "index_seconds": index_seconds, "build_seconds": perf_counter() - started,
                    "artifact_bytes": sum((temporary / name).stat().st_size for name in artifacts)}
        save(temporary / "manifest.json", metadata)
        # Publish only a complete cache, retaining prior artifacts on all failures.
        temporary.rename(directory)
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true", help="Download the pinned model snapshot before building")
    args = parser.parse_args()
    try:
        if args.download_model:
            download_model(BASELINE_PARAMETERS)
        metadata = build(load_eligible_occupations(Path("data/occupations.csv")), BASELINE_PARAMETERS)
        print(json.dumps(metadata, indent=2))
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"M4 build FAILED: {error}\n")


if __name__ == "__main__":
    main()
