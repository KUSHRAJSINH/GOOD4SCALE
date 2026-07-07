"""
server.py — FastAPI server for the BrightBox Voice AI Agent.

Two endpoints:
  POST /voice  — Twilio Voice webhook, returns TwiML that opens a Media Stream
  WS   /ws     — WebSocket endpoint for Twilio Media Streams → Pipecat pipeline

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ─── Validate required environment variables on startup ───────────────────────
REQUIRED_ENV = ["GOOGLE_API_KEY", "BASE_URL"]
for key in REQUIRED_ENV:
    if not os.getenv(key):
        raise RuntimeError(
            f"Missing required environment variable: {key}\n"
            "Copy .env.example to .env and fill in your values."
        )

BASE_URL: str = os.environ["BASE_URL"].rstrip("/")


# ─── Lifespan: pre-warm Whisper model on startup ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Server starting up — pre-loading models…")
    # Pre-warm the RAG embedding model (singleton, loads once)
    try:
        from agent.rag import query_knowledge_base
        _ = query_knowledge_base("warm up", n_results=1)
        log.info("✅ RAG / ChromaDB ready")
    except Exception as exc:
        log.warning(f"⚠️  RAG warm-up failed: {exc} — run `python scripts/ingest_kb.py` first")

    # Whisper model pre-warming is deferred to first call to avoid blocking startup
    log.info(f"✅ Server ready — public URL: {BASE_URL}")
    yield
    log.info("Server shutting down")


app = FastAPI(
    title="BrightBox Voice AI Agent",
    description="Voice support agent for BrightBox subscription box service",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "BrightBox Voice AI Agent"}


# ─── Twilio Voice Webhook ─────────────────────────────────────────────────────
@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio calls this URL when an inbound call arrives.
    We return TwiML that:
      1. Greets the caller briefly (while the WebSocket pipeline is connecting)
      2. Opens a bidirectional audio stream to our /ws endpoint
    """
    ws_url = BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    log.info(f"Inbound call — opening stream to: {ws_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}"/>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


# ─── Twilio Media Stream WebSocket ────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Twilio connects here via WebSocket (Media Streams protocol).
    We hand the WebSocket to the Pipecat pipeline which handles all
    audio I/O, STT, LLM reasoning, TTS, and call lifecycle.
    """
    await websocket.accept()
    log.info("Twilio WebSocket connection accepted")

    try:
        # Import here to defer Whisper model load until first call
        from agent.pipeline import run_brightbox_pipeline
        await run_brightbox_pipeline(websocket)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected (caller hung up)")
    except Exception as exc:
        log.error(f"Pipeline error: {exc}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ─── Entry point (for direct `python server.py` usage) ───────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
