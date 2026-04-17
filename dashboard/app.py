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
app.config["TEMPLATES_AUTO_RELOAD"] = True   # always serve fresh templates
CORS(app)

# ─────────────────────────────────────────────
# Shared state (thread-safe) — use list of queues for broadcast
# ─────────────────────────────────────────────

_client_queues: list[queue.Queue] = []  # One queue per connected SSE client
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
        
        # Broadcast to all connected clients
        dead_queues = []
        for i, q in enumerate(_client_queues):
            try:
                q.put_nowait(frame)
            except queue.Full:
                # Queue is full, try to drop old frame and retry
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except queue.Empty:
                    q.put_nowait(frame)
        
        # Debug: Log push every 50 frames
        if frame.get("t", 0) % 50 == 0:
            print(f"[SSE] Frame {frame.get('t')} broadcast to {len(_client_queues)} clients", flush=True)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────


@app.route("/")
def index():
    response = app.make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/stream")
def stream():
    """Server-Sent Events endpoint — pushes JSON frames to the browser."""
    
    # Create a new queue for this client
    client_queue: queue.Queue = queue.Queue(maxsize=50)
    
    with _lock:
        _client_queues.append(client_queue)
        client_id = len(_client_queues)
    
    def event_generator():
        heartbeat_interval = 15  
        last_heartbeat = time.time()
        frame_count = 0
        
        print(f"[SSE] Client #{client_id} connected (total clients: {len(_client_queues)})", flush=True)
        
        try:
            while True:
                try:
                    frame = client_queue.get(timeout=SSE_INTERVAL)
                    frame_count += 1
                    payload = json.dumps(frame)
                    if frame_count % 50 == 0:
                        print(f"[SSE] Client #{client_id} yielding frame #{frame_count}", flush=True)
                    yield f"data: {payload}\n\n"
                    last_heartbeat = time.time()
                except queue.Empty:
                    if time.time() - last_heartbeat > heartbeat_interval:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.time()
        finally:
            # Remove this client's queue when connection closes
            with _lock:
                try:
                    _client_queues.remove(client_queue)
                    print(f"[SSE] Client #{client_id} disconnected (remaining: {len(_client_queues)})", flush=True)
                except ValueError:
                    pass

    response = Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Encoding": "identity",
        },
    )
    return response


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
