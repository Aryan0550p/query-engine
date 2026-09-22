"""
app.py
FastAPI backend for the Intelligent Analytics Query Engine.
Auto-loads GROK_API_KEY from .env on startup.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional

# Load .env before anything else
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from query_engine import QueryEngine

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("GROQ_API_KEY", os.environ.get("GROK_API_KEY", ""))
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Intelligent Analytics Query Engine",
    description="NL -> Executable Analytics with 4-stage GenAI pipeline (Grok)",
    version="1.0.0",
)

_engine: Optional[QueryEngine] = None


def get_engine() -> QueryEngine:
    global _engine
    if _engine is None:
        if not API_KEY:
            raise HTTPException(
                status_code=503,
                detail="API key not set. Add GROK_API_KEY to .env or enter via the sidebar.",
            )
        _engine = QueryEngine(API_KEY)
    return _engine


# ─── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class FeedbackRequest(BaseModel):
    query: str
    is_correct: bool
    notes: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.post("/api/set-key")
async def set_api_key(req: ApiKeyRequest):
    global API_KEY, _engine
    API_KEY = req.api_key
    os.environ["GROK_API_KEY"] = req.api_key
    _engine = None
    return {"status": "ok", "message": "API key updated. Ready to query!"}


@app.post("/api/query")
async def run_query(req: QueryRequest):
    engine = get_engine()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.run, req.query, req.session_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    engine = get_engine()
    entry = engine.submit_feedback(req.query, req.is_correct, req.notes)
    stats = engine.get_feedback_stats()
    return {"status": "recorded", "entry": entry, "stats": stats}


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    try:
        engine = get_engine()
        return engine.get_feedback_stats()
    except HTTPException:
        return {"total": 0, "correct": 0, "incorrect": 0, "accuracy": None}


@app.get("/api/sample-queries")
async def get_sample_queries():
    engine = get_engine()
    return {"queries": engine.get_sample_queries()}


@app.post("/api/benchmark")
async def run_benchmark():
    engine = get_engine()
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, engine.run_all_benchmarks)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear-context")
async def clear_context():
    try:
        get_engine().clear_context()
    except HTTPException:
        pass
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "engine_initialized": _engine is not None,
        "api_key_set": bool(API_KEY),
        "model": "grok-3-mini (xAI)",
    }


# ─── WebSocket streaming ───────────────────────────────────────────────────────

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            query = payload.get("query", "").strip()
            session_id = payload.get("session_id", "default")

            if not query:
                await websocket.send_json({"type": "error", "message": "Empty query"})
                continue

            try:
                engine = get_engine()
            except HTTPException as e:
                await websocket.send_json({"type": "error", "message": e.detail})
                continue

            await websocket.send_json({"type": "stage_start", "stage": 1, "label": "Parsing Query DNA..."})

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, engine.run, query, session_id)

            await websocket.send_json({
                "type": "stage_complete", "stage": 1, "label": "Query DNA Parsed",
                "data": {"intent": result["intent"], "intent_explanation": result["intent_explanation"], "sub_queries": result["sub_queries"]},
            })
            await asyncio.sleep(0.05)

            await websocket.send_json({
                "type": "stage_complete", "stage": 2, "label": "Code Generated & Executed",
                "data": {"generated_logic": result["generated_logic"], "result": result["result"], "anomalies": result["anomalies"], "execution_error": result["execution_error"]},
            })
            await asyncio.sleep(0.05)

            await websocket.send_json({
                "type": "stage_complete", "stage": 3, "label": "Self-Critique Complete",
                "data": {"confidence_score": result["confidence_score"], "confidence_label": result["confidence_label"],
                         "what_was_understood": result["what_was_understood"], "how_result_was_derived": result["how_result_was_derived"],
                         "caveats": result["caveats"], "suggested_improvement": result["suggested_improvement"]},
            })
            await asyncio.sleep(0.05)

            await websocket.send_json({
                "type": "stage_complete", "stage": 4, "label": "Feedback Bias Applied",
                "data": {"feedback_bias": result["feedback_bias"], "confidence_raw": result["confidence_raw"], "confidence_adjusted": result["confidence_score"]},
            })

            await websocket.send_json({"type": "complete", "result": result})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─── Static files ──────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Frontend not found.</h1>")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  Intelligent Analytics Query Engine  (Grok / xAI)")
    print("=" * 60)
    print(f"  Open http://localhost:8000 in your browser")
    print(f"  API key loaded: {'YES' if API_KEY else 'NO'}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
