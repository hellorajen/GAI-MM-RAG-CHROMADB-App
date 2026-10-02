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

GENERATION_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2"
# Version the collection because vectors created with text-embedding-004 cannot be mixed
# with vectors created by gemini-embedding-2.
COLLECTION_NAME = "mm_rag_gemini_embedding_2"
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
    """Retrieves Google Gemini API key and automatically configures genai SDK."""
    key = ""

    # 1. Custom key entered in sidebar text input
    user_typed_key = st.session_state.get("user_custom_api_key", "").strip()
    if user_typed_key:
        key = user_typed_key

    # 2. Streamlit Cloud Secrets (hidden backend key)
    if not key:
        try:
            if (
                "GEMINI_API_KEY" in st.secrets
                and str(st.secrets["GEMINI_API_KEY"]).strip()
            ):
                key = str(st.secrets["GEMINI_API_KEY"]).strip()
            elif (
                "GOOGLE_API_KEY" in st.secrets
                and str(st.secrets["GOOGLE_API_KEY"]).strip()
            ):
                key = str(st.secrets["GOOGLE_API_KEY"]).strip()
        except Exception:
            pass

    # 3. Local Environment Variables
    if not key:
        key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()

    # Automatically configure Gemini SDK if a valid key is found
    if key:
        genai.configure(api_key=key)

    return key
def get_genai_client(api_key: str) -> genai.Client:
    """Create and return a google-genai Client targeting the stable v1 API."""

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


def embed_texts(texts, for_query=False):
    """Generates embeddings using text-embedding-004."""
    # Guarantee genai is configured before making API calls
    active_key = get_api_key()
    if not active_key:
        raise ValueError("No Gemini API key configured.")

    task_type = "retrieval_query" if for_query else "retrieval_document"

    # Ensure input is a list
    if isinstance(texts, str):
        texts = [texts]

    try:
        response = genai.embed_content(
            model="text-embedding-004", content=texts, task_type=task_type
        )
        embeddings = response["embedding"]

        # If embedding a single string, wrap in list for ChromaDB
        if len(texts) == 1 and isinstance(embeddings[0], float):
            return [embeddings]

        return embeddings
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return []

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


def index_bytes(filename: str, data: bytes) -> int:
    """Decodes upload bytes into text chunks and indexes into ChromaDB."""
    try:
        text_content = data.decode("utf-8")
    except UnicodeDecodeError:
        text_content = data.decode("latin-1", errors="ignore")

    if not text_content.strip():
        return 0

    # Break into 500-character chunks
    chunk_size = 500
    chunks = [
        text_content[i : i + chunk_size]
        for i in range(0, len(text_content), chunk_size)
    ]

    # Generate embeddings
    embeddings = embed_texts(chunks, for_query=False)

    if not embeddings:
        return 0

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas
    )

    return len(chunks)


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

        # UI Text Input: Starts empty by default to prevent secret key exposure.
        # Clicking the eye icon will show nothing unless a user types a custom key.
        st.text_input(
            "Google AI Studio API key",
            type="password",
            key="user_custom_api_key",
            placeholder="Leave blank to use default backend key",
            help="Stored in this session only. Leave blank to use the app's secure backend key.",
        )

        # Retrieve effective key for validation feedback
        active_key = get_api_key()
        if active_key:
            st.caption("🔒 Key active and loaded securely from backend secrets.")
        else:
            st.warning(
                "⚠️ No API key found. Enter a key above or configure Secrets."
            )

        st.divider()

        # Document Upload Section
        st.header("Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, MD, Images)",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        added_total = 0
        if uploaded_files:
            if not active_key:
                st.error("Please provide an API key before indexing documents.")
            else:
                with st.spinner("Indexing documents into ChromaDB..."):
                    for upload in uploaded_files:
                        data = upload.read()
                        added_total += index_bytes(upload.name, data)
                if added_total > 0:
                    st.success(f"Indexed {added_total} chunks successfully!")

        st.divider()

        # Query & Retrieval Settings
        st.header("Retrieval Settings")
        n_results = st.slider(
            "Number of context chunks to retrieve (k)",
            min_value=1,
            max_value=10,
            value=3,
        )

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
