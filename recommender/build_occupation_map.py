"""Fixed 2D occupation map for the explanation panel (SEMANTIC_LEDGER_PLAN section 10.2).

Read-only explanation support: nothing here scores, ranks or fuses occupations.

* RIASEC hexagon: computed from the catalog z-scores with a closed formula, so it needs no
  model artifact and is also computed live by the server.
* Semantic map: t-SNE (cosine, fixed seed) of the fine-tuned occupation vectors, computed once
  offline and saved as a versioned artifact with fingerprints. A stale or missing artifact only
  hides the semantic tab.

Build (refuses to overwrite an existing artifact):

    python -m recommender.build_occupation_map
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .schemas import TYPES

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "artifacts/ranking/occupation_semantic_v1/finetuned_index"
OUTPUT = ROOT / "artifacts/ranking/occupation_semantic_v1/occupation_map_v1"
CATALOG = ROOT / "data/occupations.csv"
MAP_VERSION = "occupation_map_v1"
CODE_ORDER = "eligible_code_ascending"
TSNE_SETTINGS = {"method": "tsne", "metric": "cosine", "perplexity": 30.0, "init": "pca",
                 "random_state": 0, "max_iter": 1000}

# Holland's hexagon: R, I, A, S, E, C clockwise from the top; neighbours are similar types.
HEX_CORNERS = {t: (round(math.cos(math.radians(-90 + 60 * i)), 6), round(math.sin(math.radians(-90 + 60 * i)), 6))
               for i, t in enumerate(TYPES)}


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codes_digest(codes) -> str:
    """Identifies the code order shared by map coordinates and rank snapshots."""
    return hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest()


def hexagon_point(values: dict[str, float], types=TYPES) -> list[float] | None:
    """sum_t max(0, v_t) * c_t / sum_t max(0, v_t) over `types`; None when nothing is positive."""
    weights = {t: max(0.0, float(values[t])) for t in types if t in values}
    total = sum(weights.values())
    if total <= 0:
        return None
    return [round(sum(w * HEX_CORNERS[t][0] for t, w in weights.items()) / total, 6),
            round(sum(w * HEX_CORNERS[t][1] for t, w in weights.items()) / total, 6)]


def hexagon_layout(occupations) -> dict:
    """Catalog-only hexagon positions in code order; no positive z-score sits at the centre."""
    ordered = sorted(occupations, key=lambda o: o.code)
    return {"corners": {t: list(HEX_CORNERS[t]) for t in TYPES},
            "coords": [hexagon_point(o.z_scores) or [0.0, 0.0] for o in ordered]}


def _normalise(points):
    import numpy as np
    points = np.asarray(points, dtype="float64")
    low, high = points.min(axis=0), points.max(axis=0)
    span = float((high - low).max()) or 1.0
    centred = (points - (low + high) / 2) / span   # keep the aspect ratio; fit in [-0.5, 0.5]
    return [[round(float(x), 6), round(float(y), 6)] for x, y in centred]


def tsne_layout(vectors, settings=TSNE_SETTINGS):
    from sklearn.manifold import TSNE
    params = {k: v for k, v in settings.items() if k != "method"}
    return _normalise(TSNE(n_components=2, **params).fit_transform(vectors))


def build(output: Path = OUTPUT, index: Path = INDEX, catalog: Path = CATALOG) -> Path:
    import numpy as np
    import sklearn
    from .data_loader import load_eligible_occupations

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; map artifacts are never overwritten, choose a new version")
    index_manifest = json.loads((Path(index) / "manifest.json").read_text(encoding="utf-8"))
    codes = json.loads((Path(index) / "codes.json").read_text(encoding="utf-8"))
    vectors = np.load(Path(index) / "vectors.npy", allow_pickle=False)
    occupations = load_eligible_occupations(Path(catalog))
    eligible = sorted(o.code for o in occupations)
    if codes != eligible:
        raise ValueError("index codes differ from the eligible catalog (ascending code order expected)")
    if vectors.shape[0] != len(codes):
        raise ValueError("index vectors and codes disagree")
    titles = {o.code: o.title for o in occupations}
    coords = {"version": MAP_VERSION, "code_order": CODE_ORDER, "codes_sha256": codes_digest(codes),
              "codes": codes, "titles": [titles[c] for c in codes],
              "semantic": tsne_layout(vectors), "hexagon": hexagon_layout(occupations)}
    manifest = {"version": MAP_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "code_order": CODE_ORDER, "codes_sha256": coords["codes_sha256"], "n": len(codes),
                "semantic": {**TSNE_SETTINGS, "source": "fine-tuned occupation document vectors",
                             "index": str(Path(index).resolve().relative_to(ROOT)).replace("\\", "/")
                             if Path(index).resolve().is_relative_to(ROOT) else str(index),
                             "index_files": {name: digest(Path(index) / name) for name in ("codes.json", "vectors.npy")},
                             "checkpoint_sha256": index_manifest.get("checkpoint_sha256"),
                             "index_catalog_sha256": index_manifest.get("catalog_sha256")},
                "hexagon": {"formula": "sum_t max(0, z_t) * corner_t / sum_t max(0, z_t); centre if none positive",
                            "corner_order": list(TYPES), "source": "catalog z_R..z_C"},
                "catalog_sha256": digest(catalog),
                "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
                "caveat": "t-SNE keeps neighbourhoods, not global distances."}
    output.mkdir(parents=True)
    (output / "coords.json").write_text(json.dumps(coords, ensure_ascii=False), encoding="utf-8")
    manifest["coords_sha256"] = digest(output / "coords.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def semantic_status(artifact: Path = OUTPUT, index: Path = INDEX, codes=None) -> tuple[list | None, str | None]:
    """(semantic coordinates, None) when the artifact matches the index and code order, else (None, reason)."""
    artifact = Path(artifact)
    try:
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        if digest(artifact / "coords.json") != manifest["coords_sha256"]:
            return None, "map coordinates do not match their manifest"
        semantic = manifest["semantic"]
        for name, sha in semantic["index_files"].items():
            if digest(Path(index) / name) != sha:
                return None, f"semantic index changed since the map was built ({name})"
        current = json.loads((Path(index) / "manifest.json").read_text(encoding="utf-8"))
        if current.get("checkpoint_sha256") != semantic["checkpoint_sha256"]:
            return None, "encoder checkpoint changed since the map was built"
        coords = json.loads((artifact / "coords.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"map artifact missing: build it with python -m recommender.build_occupation_map"
    except (KeyError, ValueError) as error:
        return None, f"map artifact unreadable: {error}"
    if codes is not None and coords["codes"] != list(codes):
        return None, "map code order differs from the eligible occupations"
    return coords["semantic"], None


def map_payload(occupations, artifact: Path = OUTPUT, index: Path = INDEX) -> dict:
    """What /api/occupation_map serves: hexagon always, semantic only when fresh."""
    ordered = sorted(occupations, key=lambda o: o.code)
    codes = [o.code for o in ordered]
    semantic, reason = semantic_status(artifact, index, codes)
    return {"version": MAP_VERSION, "code_order": CODE_ORDER, "codes_sha256": codes_digest(codes),
            "codes": codes, "titles": [o.title for o in ordered], "hexagon": hexagon_layout(ordered),
            "semantic": semantic, "semantic_unavailable": reason}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args(argv)
    print(build(args.output, args.index, args.catalog))


if __name__ == "__main__":
    main()
