"""Phase 7 provider wiring, against a local mock OpenAI-compatible server.

DeepSeek/Gemini keys cannot be used in tests (no network, no credit), so the
OpenAI-compatible path is exercised against a throwaway HTTP server on localhost.
This asserts the whole chain -- request shape, response parsing, guard
application, output format -- rather than just that a function is importable.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.common.config import deep_merge, load_config
from src.dataset_generation.instruction_generator import generate_dataset


class _Handler(BaseHTTPRequestHandler):
    # Populated by the fixture; lets each test control the reply.
    reply: dict = {}
    requests: list = []

    def log_message(self, *args):  # silence stderr noise
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        type(self).requests.append({"path": self.path, "body": body})
        payload = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    status = 200


@pytest.fixture()
def server():
    _Handler.requests = []
    _Handler.reply = {}
    _Handler.status = 200
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, _Handler
    srv.shutdown()


def _config(server, **over):
    srv, _ = server
    host, port = srv.server_address
    cfg = load_config()
    cfg.data = deep_merge(
        cfg.data,
        {
            "dataset_generation": {
                "provider": "openai_compatible",
                "base_url": f"http://{host}:{port}/v1",
                "model": "mock-model",
                "api_key_env": "MOCK_LLM_KEY",
                "max_chunks": 2,
                "qa_per_chunk": 2,
                "requests_per_minute": 600,
                "retry_backoff_seconds": 0,
            }
        },
    )
    cfg.data["dataset_generation"].update(over)
    return cfg


CHUNK = {
    "chunk_id": "table:T1",
    "type": "table",
    "page": 28,
    "clause_ref": "22.7",
    "caption": "Table 1 Permissible Limit for Solids",
    "markdown": "| Grade | Limit |\n| --- | --- |\n| M20 | 0.60 |",
    "embed_text": "22.7 Table 1 Permissible Limit for Solids | Grade | Limit | M20 | 0.60 |",
    "word_count": 12,
}


def test_openai_compatible_path_produces_records(server, tmp_path, monkeypatch):
    srv, handler = server
    handler.reply = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        [
                            {"instruction": "What limit applies to M20?", "input": "", "output": "0.60"},
                            {
                                "instruction": "Calculate the cement content",
                                "input": "",
                                "output": "300 kg",
                            },
                        ]
                    )
                }
            }
        ]
    }
    monkeypatch.setenv("MOCK_LLM_KEY", "test-key")
    result, records = generate_dataset_with_records(CHUNK, _config(server), tmp_path)

    # Guards working is not a fault, so a run that keeps the good pair and drops
    # the calculation pair is SUCCESS, not DEGRADED.
    assert result.status.value == "SUCCESS", result.summary_line()
    assert records, "no records written"
    # The calculation-intent question must be dropped, the other kept.
    assert all("Calculate" not in r["instruction"] for r in records)
    assert result.counts["rejected_calculation"] >= 1
    for r in records:
        assert set(r) >= {"instruction", "input", "output", "source_chunk_id"}
        assert r["source_chunk_id"] == "table:T1"

    # Request shape must be OpenAI-compatible.
    assert handler.requests, "no request reached the server"
    body = handler.requests[0]["body"]
    assert body["model"] == "mock-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    assert "Answer only from the provided text" in body["messages"][0]["content"]
    assert body["messages"][1]["role"] == "user"


def test_fenced_json_is_parsed(server, tmp_path, monkeypatch):
    srv, handler = server
    handler.reply = {
        "choices": [
            {"message": {"content": '```json\n[{"instruction":"a?","output":"0.60"}]\n```'}}
        ]
    }
    monkeypatch.setenv("MOCK_LLM_KEY", "test-key")
    result, records = generate_dataset_with_records(CHUNK, _config(server), tmp_path)
    assert records and records[0]["output"] == "0.60"


def test_ungrounded_answer_is_rejected(server, tmp_path, monkeypatch):
    srv, handler = server
    handler.reply = {
        "choices": [
            {"message": {"content": '[{"instruction":"limit?","output":"It is 999.99"}]'}}
        ]
    }
    monkeypatch.setenv("MOCK_LLM_KEY", "test-key")
    result, records = generate_dataset_with_records(CHUNK, _config(server), tmp_path)
    assert records == []
    assert result.counts["rejected_ungrounded"] == 1
    assert result.status.value == "FAILED"


def test_missing_key_is_fatal(server, tmp_path, monkeypatch):
    monkeypatch.delenv("MOCK_LLM_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result, records = generate_dataset_with_records(CHUNK, _config(server), tmp_path)
    assert result.status.value == "FAILED"
    assert "no API key" in (result.error or "") or "MOCK_LLM_KEY" in (result.error or "")


def test_402_is_quota_and_aborts_fast(server, tmp_path, monkeypatch):
    """A 402 must be fatal and must not be retried."""
    srv, handler = server
    handler.status = 402
    handler.reply = {"error": {"message": "Insufficient Balance", "type": "unknown_error"}}
    monkeypatch.setenv("MOCK_LLM_KEY", "test-key")
    result, records = generate_dataset_with_records(CHUNK, _config(server), tmp_path)
    assert result.status.value == "FAILED"
    assert "quota" in (result.error or "")
    assert len(handler.requests) == 1, "a fatal 402 must not be retried"


def test_429_rate_limit_is_retried_then_gives_up(server, tmp_path, monkeypatch):
    srv, handler = server
    handler.status = 429
    handler.reply = {"error": {"message": "Rate limit reached"}}
    monkeypatch.setenv("MOCK_LLM_KEY", "test-key")
    cfg = _config(server, max_retries=2)
    result, records = generate_dataset_with_records(CHUNK, cfg, tmp_path)
    assert result.status.value == "FAILED"
    assert len(handler.requests) == 2, "a transient 429 should be retried max_retries times"


def generate_dataset_with_records(chunk, config, tmp_path):
    """Run the phase, then read the dataset back off disk."""
    result = generate_dataset([chunk], config, tmp_path)
    out = tmp_path / "instruction_dataset.jsonl"
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
    return result, records
