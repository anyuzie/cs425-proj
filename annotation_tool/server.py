"""Isolated local annotation service; no model or ranking imports."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .contracts import Annotation
from .store import Store
from .export import export_rows, write_current_dataset


def create_app(collection_dir, annotation_dir, training_dir=None):
    training_dir = Path(training_dir) if training_dir is not None else Path(annotation_dir).parent / "training/utterances"
    store = Store(collection_dir, annotation_dir)
    app = FastAPI(title="Utterance annotation", docs_url=None, redoc_url=None)

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).parent / "index.html")

    @app.get("/api/entries")
    def entries():
        result = []
        for key, entry in store.entries().items():
            history = store.history(key)
            result.append({"id": key, "text": entry["text"], "processing_status": entry["processing_status"],
                           "created_at": entry["created_at"],
                           "saved_at": history[-1]["saved_at"] if history else None,
                           "status": history[-1]["status"] if history else "unlabelled"})
        # Newest unreviewed inputs first; the most recently saved review goes last.
        from datetime import datetime
        def timestamp(value):
            try:
                return datetime.fromisoformat(value).timestamp()
            except (TypeError, ValueError):
                return 0
        result.sort(key=lambda e: (e["status"] == "reviewed",
                    timestamp(e["saved_at"]) if e["status"] == "reviewed" else -timestamp(e["created_at"])))
        return {"entries": result, "warnings": store.warnings}

    @app.get("/api/entries/{identifier}")
    def get(identifier: str):
        try:
            return store.get(identifier)
        except KeyError:
            raise HTTPException(404, "Entry not found")
        except ValueError as error:
            raise HTTPException(400, str(error))

    @app.post("/api/entries/{identifier}")
    def save(identifier: str, annotation: Annotation):
        try:
            with store.lock:
                record = store.save(identifier, annotation)
                try:
                    path = write_current_dataset(store, training_dir)
                except (OSError, ValueError) as error:
                    raise HTTPException(503, "Annotation revision saved, but training file could not be updated. Reload and save again: " + str(error))
                return {**record, "training_path": str(path)}
        except KeyError:
            raise HTTPException(404, "Entry not found")
        except FileExistsError as error:
            raise HTTPException(409, str(error))
        except ValueError as error:
            raise HTTPException(422, str(error))

    @app.get("/api/export")
    def export():
        try:
            rows, manifest = export_rows(store)
            return {"rows": rows, "manifest": manifest}
        except ValueError as error:
            raise HTTPException(422, str(error))

    return app
