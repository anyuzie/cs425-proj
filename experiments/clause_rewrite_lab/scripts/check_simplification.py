"""Inspect the authors' released simplification checkpoint as a separate candidate."""
from pathlib import Path
import gdown
import zipfile

root = Path(__file__).resolve().parents[1] / "resources/diffuseq_ts"
root.mkdir(exist_ok=True)
path = root / "official_release"
if not path.exists():
    gdown.download(id="18poofTx14sL-kvK2g_7ekw9zp42gT6hE", output=str(path), quiet=False)
with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    print("ARCHIVE MEMBERS", names[:20], flush=True)
    if any(name.endswith("training_args.json") for name in names):
        for item in archive.infolist():
            target = (root / item.filename.replace(":", "_")).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Unsafe checkpoint archive")
            if not item.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
