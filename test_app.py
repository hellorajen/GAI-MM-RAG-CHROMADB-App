import os

from streamlit.testing.v1 import AppTest

from app import chunk_text, content_id, extract_text_file


def test_chunk_text_returns_single_chunk_for_short_input():
    assert chunk_text("hello world", chunk_size=50, overlap=5) == ["hello world"]


def test_chunk_text_splits_long_input():
    text = "Sentence number one. " * 40
    chunks = chunk_text(text, chunk_size=80, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_extract_text_file_decodes_bytes():
    assert extract_text_file(b"hello rag") == "hello rag"


def test_content_id_is_stable():
    assert content_id("a", "b") == content_id("a", "b")
    assert content_id("a", "b") != content_id("a", "c")


def test_app_renders_without_api_key():
    os.environ["CHROMA_EPHEMERAL"] = "1"
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Multimodal RAG"
    assert any("Google AI Studio API key" in box.label for box in at.sidebar.text_input)
