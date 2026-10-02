"""Multimodal RAG app using Google AI Studio (Gemini) and ChromaDB."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types as genai_types
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pypdf import PdfReader

load_dotenv()

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

GENERATION_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"
COLLECTION_NAME = "mm_rag"
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_TYPES = {".txt", ".md"}
PDF_TYPES = {".pdf"}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            split_at = cleaned.rfind(". ", start, end)
            if split_at <= start:
                split_at = cleaned.rfind(" ", start, end)
            if split_at > start:
                end = split_at + 1
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_text_file(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()


def content_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def _secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value or "")


def get_api_key() -> str:
    keyed = st.session_state.get("google_api_key", "")
    if keyed:
        return str(keyed)
    return _secret("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


def get_genai_client(api_key: str) -> genai.Client:
    """Create and return a google-genai Client targeting the stable v1 API.

    The SDK defaults to v1beta where text-embedding-004 was removed.
    Forcing api_version='v1' ensures the correct endpoint is used.
    """
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(api_version="v1"),
    )


@st.cache_resource
def get_chroma_client() -> Any:
    if os.environ.get("CHROMA_EPHEMERAL") == "1":
        return chromadb.EphemeralClient()
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection() -> Any:
    return get_chroma_client().get_or_create_collection(name=COLLECTION_NAME)


def _find_embedding_model(client: genai.Client) -> str:
    """Return the best available embedding model name for this API key.

    Tries a prioritised list and picks the first one that supports
    embedContent. Falls back to 'text-embedding-004' as a last resort so
    the error message remains informative if nothing is found.
    """
    preferred = [
        "text-embedding-004",
        "text-embedding-005",
        "embedding-001",
    ]
    try:
        available = {
            m.name.split("/")[-1]  # strip 'models/' prefix
            for m in client.models.list()
            if "embedContent" in (m.supported_actions or [])
        }
        for name in preferred:
            if name in available:
                return name
    except Exception:
        pass  # If listing fails, fall through to the hardcoded default
    return "text-embedding-004"


def embed_texts(texts, api_key: str, for_query: bool = False) -> list:
    """Embed a list of texts using the best available Gemini embedding model.

    Uses the new google-genai SDK (v1 API) which supports text-embedding-004.
    Always returns a list of embedding vectors.
    """
    task_type = "RETRIEVAL_QUERY" if for_query else "RETRIEVAL_DOCUMENT"
    client = get_genai_client(api_key)
    model_name = _find_embedding_model(client)

    # Normalise input: the new SDK embed_content accepts a single string.
    # For a batch we call it once per item (SDK does not batch in one call).
    if isinstance(texts, list):
        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model=model_name,
                contents=text,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    # Single string input.
    response = client.models.embed_content(
        model=model_name,
        contents=texts,
        config=genai_types.EmbedContentConfig(task_type=task_type),
    )
    return [response.embeddings[0].values]


def caption_image(image: Image.Image, api_key: str) -> str:
    """Generate a detailed text caption for an image using Gemini."""
    client = get_genai_client(api_key)
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=[
            "Describe this image in detail for later retrieval. "
            "Include visible text, objects, charts, and layout.",
            image,
        ],
    )
    return (response.text or "").strip()


def generate_answer(
    question: str,
    contexts: list[str],
    images: list[Image.Image],
    api_key: str,
) -> str:
    """Generate an answer grounded in retrieved context and optional images."""
    context_block = "\n\n".join(
        f"[Source {i}] {chunk}" for i, chunk in enumerate(contexts, start=1)
    ) or "(no retrieved context)"
    prompt = (
        "Answer using the retrieved context and any attached images. "
        "If the context is insufficient, say so. Cite sources like [Source 1].\n\n"
        f"Context:\n{context_block}\n\nQuestion:\n{question}"
    )
    parts: list[Any] = [prompt, *images]
    client = get_genai_client(api_key)
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=parts,
    )
    return (response.text or "").strip()


def index_bytes(filename: str, data: bytes, api_key: str) -> int:
    """Index a single file into ChromaDB, chunking and embedding its content."""
    suffix = Path(filename).suffix.lower()
    collection = get_collection()
    added = 0

    if suffix in PDF_TYPES:
        text = extract_pdf_text(data)
        kind = "pdf"
        chunks = chunk_text(text)
        extra_docs: list[tuple[str, dict[str, str]]] = [
            (chunk, {"filename": filename, "type": kind, "chunk": str(i)})
            for i, chunk in enumerate(chunks)
        ]
    elif suffix in TEXT_TYPES:
        text = extract_text_file(data)
        kind = "text"
        chunks = chunk_text(text)
        extra_docs = [
            (chunk, {"filename": filename, "type": kind, "chunk": str(i)})
            for i, chunk in enumerate(chunks)
        ]
    elif suffix in IMAGE_TYPES:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        description = caption_image(image, api_key)
        extra_docs = [
            (
                f"Image {filename}: {description}",
                {"filename": filename, "type": "image", "chunk": "0"},
            )
        ]
    else:
        raise ValueError(f"Unsupported file type: {suffix or filename}")

    if not extra_docs:
        return 0

    documents = [doc for doc, _ in extra_docs]
    metadatas = [meta for _, meta in extra_docs]
    ids = [content_id(filename, meta["chunk"], doc) for doc, meta in extra_docs]
    embeddings = embed_texts(documents, api_key=api_key, for_query=False)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    added += len(documents)
    return added


def retrieve(
    query: str, n_results: int, api_key: str
) -> tuple[list[str], list[dict[str, str]]]:
    """Query ChromaDB with a semantic embedding of ``query``."""
    collection = get_collection()
    if collection.count() == 0:
        return [], []
    query_embedding = embed_texts([query], api_key=api_key, for_query=True)
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    return list(documents), [dict(meta) for meta in metadatas]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("google_api_key", get_api_key())


def render_sidebar() -> int:
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Google AI Studio API key",
            type="password",
            value=st.session_state.google_api_key,
            help="Stored in this session only. You can also set GOOGLE_API_KEY or .streamlit/secrets.toml.",
        )
        st.session_state.google_api_key = api_key

        n_results = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

        st.divider()
        st.subheader("Ingest files")
        uploads = st.file_uploader(
            "PDF, text, or images",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=True,
        )
        ingest = st.button("Add to ChromaDB", type="primary", icon=":material/upload:")

        if ingest:
            if not api_key:
                st.error("Add an API key before ingesting files.")
            elif not uploads:
                st.warning("Choose at least one file.")
            else:
                added_total = 0
                with st.status("Indexing files", expanded=True) as status:
                    for upload in uploads:
                        data = upload.getvalue()
                        st.write(f"Indexing {upload.name}")
                        added_total += index_bytes(upload.name, data, api_key)
                    status.update(
                        label=f"Indexed {added_total} chunk(s)",
                        state="complete",
                    )
                st.success(f"Added {added_total} chunk(s) to the collection.")
                st.rerun()

        collection = get_collection()
        st.caption(f"{collection.count()} chunk(s) in `{COLLECTION_NAME}`.")
        if st.button("Clear collection", icon=":material/delete:"):
            get_chroma_client().delete_collection(COLLECTION_NAME)
            get_chroma_client.clear()
            st.rerun()

    return n_results


def _images_from_chat_files(files: list[Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for file in files:
        name = getattr(file, "name", "") or ""
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_TYPES:
            continue
        images.append(Image.open(io.BytesIO(file.getvalue())).convert("RGB"))
    return images


def run_query(question: str, n_results: int, images: list[Image.Image]) -> None:
    api_key = get_api_key()
    if not api_key:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Add a Google AI Studio API key in the sidebar to ask questions.",
                "sources": [],
            }
        )
        return

    contexts, metadatas = retrieve(question, n_results=n_results, api_key=api_key)
    answer = generate_answer(question, contexts, images, api_key=api_key)
    sources = [
        {
            "filename": meta.get("filename", "unknown"),
            "type": meta.get("type", ""),
            "text": context,
        }
        for context, meta in zip(contexts, metadatas, strict=False)
    ]
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main() -> None:
    st.set_page_config(
        page_title="Multimodal RAG",
        page_icon=":material/auto_awesome:",
        layout="centered",
    )
    init_state()
    n_results = render_sidebar()

    st.title("Multimodal RAG")
    st.caption("Ask questions over PDFs, text, and images stored in ChromaDB, using Gemini.")

    if not st.session_state.messages:
        st.info("Upload files in the sidebar, then ask a question below.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources") or []
            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(f"**{source['filename']}** ({source['type']})")
                        st.caption(source["text"])

    prompt = st.chat_input(
        "Ask about your indexed files",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"],
        submit_mode="disable",
    )

    if prompt:
        if isinstance(prompt, str):
            question = prompt
            files: list[Any] = []
        else:
            question = (prompt.text or "").strip()
            files = list(prompt.files or [])

        images = _images_from_chat_files(files)
        extra_notes: list[str] = []
        api_key = get_api_key()
        if files and api_key:
            for file in files:
                extra_notes.append(f"Attached {file.name}")
                index_bytes(file.name, file.getvalue(), api_key)
        elif files and not api_key:
            extra_notes.append("Attachments were not indexed because no API key is set.")

        user_text = question or "(see attached files)"
        if extra_notes:
            user_text = user_text + "\n\n" + "\n".join(extra_notes)

        st.session_state.messages.append({"role": "user", "content": user_text, "sources": []})
        run_query(question or "Describe the attached files.", n_results, images)
        st.rerun()


if __name__ == "__main__":
    main()
