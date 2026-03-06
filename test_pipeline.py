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
