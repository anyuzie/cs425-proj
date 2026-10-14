import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pipeline.interfaces import MAX_TURNS, State
from pipeline.system import System, TurnResult

from . import turn_log
from .schemas import ChatRequest, Health, SessionRequest, TurnResponse

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Session:
    state: State
    participant_code: str
    debug: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.system = System.load(os.environ.get("CM_DATA_DIR", ROOT / "data"))
    app.state.sessions = {}
    app.state.lock = threading.Lock()
    yield


app = FastAPI(title="Career matcher", lifespan=lifespan)


def _study_mode() -> bool:
    # Human-study sessions must never show participants their own profile.
    return os.environ.get("CM_STUDY_MODE") == "1"


def _log_dir() -> Path:
    return Path(os.environ.get("CM_LOG_DIR", ROOT / "logs"))


def _response(sid: str, s: Session, r: TurnResult, system: System) -> dict:
    return {"session_id": sid, "participant_code": s.participant_code, "reply": r.reply, "action": r.action,
            "turn": r.turn, "max_turns": MAX_TURNS, "done": r.done, "recommendations": r.recommendations,
            "trace": system.trace(s.state) if s.debug else None, "quotes": r.quotes}


def _log(system: System, sid: str, s: Session, text: str | None, out: dict) -> None:
    turn_log.append(_log_dir(), s.participant_code, {
        "session_id": sid, "turn": out["turn"], "user_text": text, "reply": out["reply"],
        "action": out["action"], "quotes": out["quotes"], "recommendations": out["recommendations"],
        "done": out["done"], "profile": system.profile(s.state), "debug": s.debug})


@app.get("/api/health", response_model=Health)
def health():
    system = app.state.system
    return {"status": "ok", "eligible": system.eligible, "components": system.components}


@app.post("/api/session", response_model=TurnResponse)
def create_session(req: SessionRequest):
    system = app.state.system
    s = Session(State(), req.participant_code or f"anon-{uuid.uuid4().hex[:8]}",
                req.debug and not _study_mode())
    sid = uuid.uuid4().hex
    with app.state.lock:
        app.state.sessions[sid] = s
        out = _response(sid, s, system.start(s.state), system)
        _log(system, sid, s, None, out)
    return out


@app.post("/api/chat", response_model=TurnResponse)
def chat(req: ChatRequest):
    system = app.state.system
    with app.state.lock:
        s = app.state.sessions.get(req.session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        if s.state.done:
            raise HTTPException(409, "conversation is finished")
        out = _response(req.session_id, s, system.step(s.state, req.text), system)
        _log(system, req.session_id, s, req.text, out)
    return out


# Mounted last so it cannot shadow /api. check_dir=False lets the API run without the frontend.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True, check_dir=False))
