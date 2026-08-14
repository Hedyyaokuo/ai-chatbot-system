from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, request
from flask_cors import CORS

from agent import KNOWLEDGE_MANIFEST, run_cloud_agent


app = Flask(__name__)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://hedyyaokuo.github.io,http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

request_times: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(client_id: str) -> bool:
    now = time.time()
    window = request_times[client_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= 12:
        return True
    window.append(now)
    return False


def _serialise_source(document: dict) -> dict:
    return {
        "source_file": document.get("source_file", "原始知识库"),
        "source_path": document.get("source_path", ""),
        "modality": document.get("modality", "text"),
        "section": document.get("section", "general"),
        "document_family": document.get("document_family", "general"),
        "page_label": document.get("page_label", ""),
        "chunk_id": document.get("chunk_id"),
        "retrieval_score": document.get("retrieval_score"),
        "retrieval_tool": document.get("retrieval_tool", ""),
        "content": document.get("content", "")[:500],
    }


def _read_payload() -> dict:
    if request.is_json:
        return request.get_json(silent=True) or {}
    try:
        return json.loads(request.get_data(as_text=True) or "{}")
    except json.JSONDecodeError:
        return {}


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "service": "yixin-original-personalised-multimodal-agent",
        "mode": "groq" if os.getenv("GROQ_API_KEY") else "extractive-fallback",
        "knowledge_base": {
            "records": KNOWLEDGE_MANIFEST["records"],
            "text_chunks": KNOWLEDGE_MANIFEST["text_chunks"],
            "image_captions": KNOWLEDGE_MANIFEST["image_captions"],
            "source_files": KNOWLEDGE_MANIFEST["source_files"],
            "chunk_size": KNOWLEDGE_MANIFEST["chunk_size"],
            "chunk_overlap": KNOWLEDGE_MANIFEST["chunk_overlap"],
        },
    })


@app.post("/api/chat")
def chat():
    client_id = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    if _rate_limited(client_id.split(",")[0].strip()):
        return jsonify({"ok": False, "error": "请求过于频繁，请稍后再试。"}), 429

    payload = _read_payload()
    message = str(payload.get("message", "")).strip()
    session_id = str(payload.get("session_id", "public-session")).strip()

    if not message:
        return jsonify({"ok": False, "error": "请输入问题。"}), 400
    if len(message) > 2000:
        return jsonify({"ok": False, "error": "单次问题不能超过 2000 个字符。"}), 400
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", session_id):
        return jsonify({"ok": False, "error": "无效的会话编号。"}), 400

    try:
        result = run_cloud_agent(message, session_id)
        return jsonify({
            "ok": True,
            "answer": result.get("answer", ""),
            "query_family": result.get("query_family", "general"),
            "selected_tool": result.get("selected_tool", ""),
            "verification_result": result.get("verification_result", ""),
            "trace": result.get("trace", []),
            "sources": [
                _serialise_source(document)
                for document in result.get("retrieved_docs", [])
            ],
        })
    except Exception as exc:
        app.logger.exception("Cloud agent request failed")
        return jsonify({
            "ok": False,
            "error": f"智能体暂时无法回答：{exc}",
        }), 500


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "message": "AI Chatbot API is running.",
        "frontend": "https://hedyyaokuo.github.io/ai-chatbot-system/",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
