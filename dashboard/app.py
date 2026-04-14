"""
app.py — Flask server with Server-Sent Events (SSE) for the live 5G dashboard.

Routes:
    GET /          → serves index.html
    GET /stream    → SSE stream of JSON frames pushed by the pipeline
    POST /push     → pipeline writes data here (internal)
"""

import json
import queue
import threading
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, Response, render_template, request, jsonify
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, SSE_INTERVAL, REWARD_WINDOW

app = Flask(__name__, template_folder="templates")
CORS(app)

# ─────────────────────────────────────────────
# Shared state (thread-safe)
# ─────────────────────────────────────────────

_data_queue: queue.Queue = queue.Queue(maxsize=200)
_last_frame: dict = {}
_reward_history: list = []
_sla_history: list = []
_lock = threading.Lock()


def push_frame(frame: dict):
    """Called by the pipeline to enqueue a new data frame."""
    global _last_frame, _reward_history, _sla_history
    with _lock:
        _last_frame = frame
        _reward_history.append(frame.get("reward", 0.0))
        _sla_history.append(frame.get("sla_violations", 0))
        if len(_reward_history) > 2000:
            _reward_history = _reward_history[-2000:]
        if len(_sla_history) > 2000:
            _sla_history = _sla_history[-2000:]
    try:
        _data_queue.put_nowait(frame)
    except queue.Full:
        try:
            _data_queue.get_nowait()  # drop oldest
        except queue.Empty:
            pass
        _data_queue.put_nowait(frame)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes JSON frames to the browser."""

    def event_generator():
        heartbeat_interval = 15  # send heartbeat every N seconds to keep alive
        last_heartbeat = time.time()

        while True:
            try:
                frame = _data_queue.get(timeout=SSE_INTERVAL)
                payload = json.dumps(frame)
                yield f"data: {payload}\n\n"
                last_heartbeat = time.time()
            except queue.Empty:
                # Heartbeat to prevent browser from closing the connection
                if time.time() - last_heartbeat > heartbeat_interval:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.time()

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/history")
def history():
    """Return rolling reward and SLA violation history for initial chart render."""
    with _lock:
        return jsonify(
            {
                "reward_history": _reward_history[-500:],
                "sla_history": _sla_history[-500:],
                "last_frame": _last_frame,
            }
        )


@app.route("/push", methods=["POST"])
def push():
    """Internal endpoint for the pipeline to push data."""
    data = request.get_json(force=True)
    push_frame(data)
    return jsonify({"ok": True})


# ─────────────────────────────────────────────
# Dev server launcher
# ─────────────────────────────────────────────


def run_server(host: str = FLASK_HOST, port: int = FLASK_PORT, debug: bool = False):
    """Launch Flask in a background daemon thread."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=debug, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"Dashboard running at http://localhost:{port}")
    return t
