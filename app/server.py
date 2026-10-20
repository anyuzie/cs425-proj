import copy
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pipeline.interfaces import State
from pipeline.system import System, TurnResult

from . import turn_log
from .schemas import ChatRequest, Health, SessionRequest, TurnResponse

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("app.server")


@dataclass
class Session:
    state: State
    participant_code: str
    debug: bool
    lock: threading.Lock = field(default_factory=threading.Lock)  # held during step + log
    last_message_id: str | None = None
    last_response: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.system = System.load(os.environ.get("CM_DATA_DIR", ROOT / "data"),
                                  os.environ.get("CM_CONFIG", ROOT / "configs" / "system.json"))
    app.state.sessions = {}
    app.state.lock = threading.Lock()  # guards only the sessions dict lookup/insert
    yield


app = FastAPI(title="Career matcher", lifespan=lifespan)


def _study_mode() -> bool:
    # Human-study sessions must never show participants their own profile.
    return os.environ.get("CM_STUDY_MODE") == "1"


def _log_dir() -> Path:
    return Path(os.environ.get("CM_LOG_DIR", ROOT / "logs"))


def _response(sid: str, s: Session, state: State, r: TurnResult, system: System) -> dict:
    return {"session_id": sid, "participant_code": s.participant_code, "reply": r.reply, "action": r.action,
            "turn": r.turn, "max_turns": system.max_turns, "done": r.done, "recommendations": r.recommendations,
            "trace": system.trace(state) if s.debug else None, "quotes": r.quotes}


def _log(system: System, sid: str, s: Session, state: State, text: str | None, out: dict) -> None:
    turn_log.append(_log_dir(), s.participant_code, {
        "session_id": sid, "turn": out["turn"], "user_text": text, "reply": out["reply"],
        "action": out["action"], "quotes": out["quotes"], "recommendations": out["recommendations"],
        "done": out["done"], "profile": system.profile(state), "debug": s.debug,
        "input_mode": system.input_mode, "components": system.components,
        "ranking_input": state.ranking_trace[-1] if state.ranking_trace else None,
        "resolved_ranking": getattr(system.ranker, "resolved_config", None)})


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
    out = _response(sid, s, s.state, system.start(s.state), system)
    _log(system, sid, s, s.state, None, out)
    with app.state.lock:
        app.state.sessions[sid] = s
    return out


@app.post("/api/chat", response_model=TurnResponse)
def chat(req: ChatRequest):
    system = app.state.system
    with app.state.lock:
        s = app.state.sessions.get(req.session_id)
    if s is None:
        raise HTTPException(404, "unknown session")
    with s.lock:
        if req.message_id and req.message_id == s.last_message_id:
            return s.last_response  # client retry of a turn that already ran
        if s.state.done:
            raise HTTPException(409, "conversation is finished")
        try:
            state = copy.deepcopy(s.state)  # commit only after the log write succeeds
            out = _response(req.session_id, s, state, system.step(state, req.text), system)
            _log(system, req.session_id, s, state, req.text, out)
        except Exception:
            log.exception("turn failed; session state left unchanged")
            raise HTTPException(500, "turn failed; please retry")
        s.state, s.last_message_id, s.last_response = state, req.message_id, out
    return out


# Mounted last so it cannot shadow /api. check_dir=False lets the API run without the frontend.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True, check_dir=False))
