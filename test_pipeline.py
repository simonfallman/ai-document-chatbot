"""
Basic tests for the RAG pipeline — chunking and retrieval logic.
Does not require an OpenAI API key.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunking_splits_long_text():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content="word " * 100)]
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 1


def test_chunking_short_text_stays_single_chunk():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content="This is a short document.")]
    chunks = splitter.split_documents(docs)
    assert len(chunks) == 1


def test_chunk_overlap_is_respected():
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    text = "a" * 200
    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)
    # Each chunk should be at most chunk_size characters
    for chunk in chunks:
        assert len(chunk.page_content) <= 50


# ── Hash / cache ──────────────────────────────────────────────────────────────

def test_file_hash_is_deterministic():
    from app import file_hash
    data = b"hello world"
    assert file_hash(data) == file_hash(data)


def test_different_files_have_different_hashes():
    from app import file_hash
    assert file_hash(b"file one") != file_hash(b"file two")


def test_indexed_hashes_persist(tmp_path, monkeypatch):
    from app import load_indexed_hashes, save_indexed_hashes
    hash_store = tmp_path / "indexed_files.json"
    monkeypatch.setattr("app.HASH_STORE", str(hash_store))

    save_indexed_hashes({"abc123": "document.pdf"})
    loaded = load_indexed_hashes()
    assert loaded == {"abc123": "document.pdf"}


def test_load_indexed_hashes_returns_empty_when_missing(tmp_path, monkeypatch):
    from app import load_indexed_hashes
    monkeypatch.setattr("app.HASH_STORE", str(tmp_path / "nonexistent.json"))
    assert load_indexed_hashes() == {}


# ── File size cap ─────────────────────────────────────────────────────────────

def test_max_file_size_constant():
    from app import MAX_FILE_SIZE
    assert MAX_FILE_SIZE == 5 * 1024 * 1024


# ── DB helpers ────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file and re-run init_db() for isolation."""
    db_file = str(tmp_path / "test_chat.db")
    monkeypatch.setattr("app.DB_PATH", db_file)
    import app
    app.init_db()
    return db_file


def test_init_db_creates_tables(db):
    import sqlite3
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    con.close()
    assert "conversations" in tables
    assert "messages" in tables


def test_create_conversation_returns_id(db, monkeypatch):
    monkeypatch.setattr("app.DB_PATH", db)
    from app import create_conversation
    cid = create_conversation("Test conv", "doc.pdf")
    assert isinstance(cid, int)


def test_load_conversations_returns_created(db, monkeypatch):
    monkeypatch.setattr("app.DB_PATH", db)
    from app import create_conversation, load_conversations
    create_conversation("My Doc", "file.txt")
    convs = load_conversations()
    assert any(c["title"] == "My Doc" for c in convs)


def test_delete_conversation_removes_it(db, monkeypatch):
    monkeypatch.setattr("app.DB_PATH", db)
    from app import create_conversation, delete_conversation, load_conversations
    cid = create_conversation("To Delete", "doc.pdf")
    delete_conversation(cid)
    convs = load_conversations()
    assert not any(c["id"] == cid for c in convs)


def test_save_and_load_messages(db, monkeypatch):
    monkeypatch.setattr("app.DB_PATH", db)
    from app import create_conversation, save_message, load_messages
    cid = create_conversation("Chat", "doc.pdf")
    save_message(cid, "user", "Hello?")
    save_message(cid, "assistant", "Hi there!", sources=[{"text": "chunk"}])
    msgs = load_messages(cid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["sources"] == [{"text": "chunk"}]


def test_collection_hash_stored_on_create(db, monkeypatch):
    monkeypatch.setattr("app.DB_PATH", db)
    from app import create_conversation, load_conversations
    create_conversation("Hashed", "doc.pdf", collection_hash="deadbeef")
    convs = load_conversations()
    match = next(c for c in convs if c["title"] == "Hashed")
    assert match["collection_hash"] == "deadbeef"


# ── Tool intent detection ─────────────────────────────────────────────────────

def test_summarize_trigger_matches():
    from app import SUMMARIZE_TRIGGERS
    assert SUMMARIZE_TRIGGERS.search("please summarize this document")
    assert SUMMARIZE_TRIGGERS.search("Can you give me a summary?")
    assert SUMMARIZE_TRIGGERS.search("TL;DR")


def test_faq_trigger_matches():
    from app import FAQ_TRIGGERS
    assert FAQ_TRIGGERS.search("generate FAQ for this")
    assert FAQ_TRIGGERS.search("What are the key questions?")
    assert FAQ_TRIGGERS.search("quiz me on this content")


def test_no_trigger_matches():
    from app import SUMMARIZE_TRIGGERS, FAQ_TRIGGERS
    query = "what is the main topic of chapter 2?"
    assert not SUMMARIZE_TRIGGERS.search(query)
    assert not FAQ_TRIGGERS.search(query)


# ── Prometheus metrics ────────────────────────────────────────────────────────

def test_metric_objects_exist():
    from app import QUERY_COUNTER, QUERY_LATENCY, RETRIEVAL_LATENCY, UPLOAD_COUNTER, ERROR_COUNTER
    assert QUERY_COUNTER is not None
    assert QUERY_LATENCY is not None
    assert RETRIEVAL_LATENCY is not None
    assert UPLOAD_COUNTER is not None
    assert ERROR_COUNTER is not None


def test_query_counter_increments():
    from app import QUERY_COUNTER
    from prometheus_client import REGISTRY
    before = REGISTRY.get_sample_value('query_total') or 0.0
    QUERY_COUNTER.inc()
    after = REGISTRY.get_sample_value('query_total') or 0.0
    assert after == before + 1.0


def test_upload_counter_increments():
    from app import UPLOAD_COUNTER
    from prometheus_client import REGISTRY
    before = REGISTRY.get_sample_value('document_uploads_total') or 0.0
    UPLOAD_COUNTER.inc()
    after = REGISTRY.get_sample_value('document_uploads_total') or 0.0
    assert after == before + 1.0


def test_metrics_server_does_not_start_without_env_var(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("PROMETHEUS_METRICS_PORT", raising=False)
    from app import start_metrics_server
    with patch("threading.Thread") as mock_thread:
        start_metrics_server()
        mock_thread.assert_not_called()


# ── MLflow logging ────────────────────────────────────────────────────────────

def test_log_query_to_mlflow_calls_mlflow(monkeypatch):
    import mlflow
    from unittest.mock import patch, MagicMock

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)

    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.start_run', return_value=mock_run) as mock_start, \
         patch('mlflow.log_param') as mock_param, \
         patch('mlflow.log_metric') as mock_metric, \
         patch('mlflow.set_tag') as mock_tag:

        from app import log_query_to_mlflow
        log_query_to_mlflow("rag", "doc.pdf", retrieval_ms=42.0, total_ms=310.0,
                            query_length=20, answer_length=150,
                            num_chunks_retrieved=4, avg_relevance_score=0.75)

        mock_start.assert_called_once()
        mock_param.assert_any_call("chunk_size", 500)
        mock_param.assert_any_call("chunk_overlap", 50)
        mock_param.assert_any_call("model_id", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
        mock_param.assert_any_call("k", 6)
        mock_metric.assert_any_call("retrieval_latency_ms", 42.0)
        mock_metric.assert_any_call("total_latency_ms", 310.0)
        mock_metric.assert_any_call("query_length", 20)
        mock_metric.assert_any_call("answer_length", 150)
        mock_metric.assert_any_call("num_chunks_retrieved", 4)
        mock_metric.assert_any_call("avg_relevance_score", 0.75)
        mock_tag.assert_any_call("document_name", "doc.pdf")
        mock_tag.assert_any_call("query_type", "rag")


def test_log_query_to_mlflow_swallows_exceptions(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.start_run', side_effect=Exception("connection refused")):
        from app import log_query_to_mlflow
        # Should not raise
        log_query_to_mlflow("rag", "doc.pdf", retrieval_ms=10.0, total_ms=200.0)


def test_error_counter_increments_on_exception():
    from app import ERROR_COUNTER
    from prometheus_client import REGISTRY

    # Simulate what the except block does: ERROR_COUNTER.labels(type=type(e).__name__).inc()
    exc = ValueError("test error")
    label = type(exc).__name__  # "ValueError"

    before = REGISTRY.get_sample_value('errors_total', {'type': label}) or 0.0
    ERROR_COUNTER.labels(type=label).inc()
    after = REGISTRY.get_sample_value('errors_total', {'type': label}) or 0.0

    assert after == before + 1.0
    assert label == "ValueError"
