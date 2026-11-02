"""Explicit preparation of real pretrained resources, separate from runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile
import io
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import requests
    import gdown
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffuseq-only", action="store_true")
    args = parser.parse_args()
    resources = ROOT / "resources"
    resources.mkdir(exist_ok=True)
    vendor = ROOT / "vendor/DiffuSeq"
    if not vendor.exists():
        response = requests.get("https://api.github.com/repos/Shark-NLP/DiffuSeq/commits/main", timeout=60)
        response.raise_for_status()
        revision = response.json()["sha"]
        response = requests.get(f"https://codeload.github.com/Shark-NLP/DiffuSeq/zip/{revision}", timeout=120)
        response.raise_for_status()
        vendor.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for item in archive.infolist():
                relative = Path(*Path(item.filename).parts[1:])
                target = (vendor / relative).resolve()
                if not target.is_relative_to(vendor.resolve()):
                    raise ValueError("Unsafe archive member")
                if item.is_dir() or not (relative.suffix in {".py", ".json", ".md"} or relative.name == "LICENSE"):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
        (vendor / "upstream_revision.json").write_text(json.dumps({"repository": "Shark-NLP/DiffuSeq", "revision": revision}, indent=2))
    checkpoint_dir = resources / "diffuseq_qqp"
    checkpoint_dir.mkdir(exist_ok=True)
    if not list(checkpoint_dir.rglob("*.pt")):
        url = "https://drive.google.com/drive/folders/1vnhJIUqPQva_x_sH2h5a0moCc1NYmEpr?usp=sharing"
        listing = gdown.download_folder(url=url, output=str(checkpoint_dir), quiet=False,
                                        remaining_ok=True, skip_download=True)
        print("DIFFUSEQ_FILES", listing, flush=True)
        # Official folder contains sample outputs too; retrieve model/config only.
        if not listing:
            raise RuntimeError("Official DiffuSeq checkpoint listing unavailable")
        selected = [entry for entry in listing if str(entry.path).endswith((".pt", "training_args.json", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"))
                    or (str(entry.path).endswith(".zip") and "samples" not in str(entry.path))]
        for entry in selected:
            target = Path(entry.local_path)
            if not target.is_absolute():
                target = ROOT / target
            if not target.resolve().is_relative_to(checkpoint_dir.resolve()):
                raise ValueError("Unexpected checkpoint path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                gdown.download(id=entry.id, output=str(target), quiet=False)
            if target.suffix == ".zip":
                with zipfile.ZipFile(target) as archive:
                    for item in archive.infolist():
                        # The official Linux archive includes ':' in its directory name.
                        destination = (checkpoint_dir / item.filename.replace(":", "_")).resolve()
                        if not destination.is_relative_to(checkpoint_dir.resolve()):
                            raise ValueError("Unsafe checkpoint archive")
                        if not item.is_dir():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(archive.read(item))
        if not list(checkpoint_dir.rglob("*.pt")):
            raise RuntimeError("Official folder did not provide a DiffuSeq checkpoint")
    if not args.diffuseq_only:
        api = HfApi()
        for name, model_id in [("flan_t5_base", "google/flan-t5-base"), ("colbertv2", "colbert-ir/colbertv2.0"), ("bert_tokenizer", "google-bert/bert-base-uncased")]:
            destination = resources / name
            identity_path = destination / "resource_identity.json"
            revision = json.loads(identity_path.read_text())["revision"] if identity_path.exists() else api.model_info(model_id).sha
            snapshot_download(model_id, revision=revision, local_dir=destination,
                              allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
                              ignore_patterns=["onnx/*", "openvino/*", *(["*.safetensors"] if name == "bert_tokenizer" else [])])
            (destination / "resource_identity.json").write_text(json.dumps({"model_id": model_id, "revision": revision}, indent=2))
    if not args.diffuseq_only:
        subprocess.run([sys.executable, str(ROOT / "scripts/check_simplification.py")], check=True)
    manifest = {str(p.relative_to(ROOT)): digest(p) for directory in (resources, vendor)
                for p in sorted(directory.rglob("*")) if p.is_file() and ".cache" not in p.parts
                and "__pycache__" not in p.parts and p != resources / "manifest.json"}
    (resources / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared", len(manifest), "resource files", flush=True)


if __name__ == "__main__":
    main()
