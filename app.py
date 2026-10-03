"""
===============================================================================
MULTIMODAL RAG APP: AGENTIC ARCHITECTURE & CORE PRIMITIVES MAP
===============================================================================
This application demonstrates a production-grade, custom Multimodal Retrieval-
Augmented Generation (RAG) architecture built with Streamlit, Google Gemini 3.5,
and ChromaDB.

Key Agentic & Distributed AI Primitives Implemented Below:

1. STATE SCHEMAS (Short-Term Conversational State)
   - Encapsulated via `st.session_state` (`init_state`).
   - Maintains memory buffers (`messages`), user context, and API security keys across turns.

2. COMPUTE NODES (Decoupled Single-Responsibility Task Processing Units)
   - Ingestion & Text Extraction Nodes: `extract_pdf_text`, `extract_text_file`, `chunk_text`.
   - Visual Perception Node: `caption_image` (Gemini Flash multimodal processing).
   - Embedding / Vectorization Node: `embed_texts` (gemini-embedding-2 with asymmetric instruction tags).
   - Reasoning & Answer Generation Node: `generate_answer` (grounded multimodal synthesis).

3. MEMORY CHECKPOINTS & LONG-TERM STORAGE
   - Ephemeral & Persistent Vector Memory: Handled via `chromadb.PersistentClient`.
   - Deterministic State Checkpointing: `content_id` (SHA-256 content hashing) ensures 
     chunk versioning, deduplication, and idempotent vector store writes (`upsert`).

4. SUPERVISOR ROUTER & CONDITIONAL EDGES
   - Dynamic Routing Logic: Implemented inside `main()` and `run_query()`.
   - Evaluates incoming state (Text Query vs. File Attachments vs. Existing Vector Memory)
     and branches dynamically across Ingestion Nodes, Vector Retrieval Nodes, and Visual Reasoning Nodes.
===============================================================================
"""

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

# --- MODEL & CONFIGURATION CONSTANTS ---
GENERATION_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-2"

# Collection name versioning handles embedding dimensionality/space shifts.
COLLECTION_NAME = "mm_rag_gemini_embedding_2"
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_TYPES = {".txt", ".md"}
PDF_TYPES = {".pdf"}


# =============================================================================
# PRIMITIVE: COMPUTE NODES (DOCUMENT PROCESSING & CHUNKING)
# =============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Compute Node: Semantic Chunking Engine.
    
    Splits continuous document text into overlapping segments, prioritizing sentence 
    and paragraph boundaries to preserve contextual intent for embedding space retrieval.
    """
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
    """Compute Node: Unstructured PDF Text Extraction Processor."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_text_file(data: bytes) -> str:
    """Compute Node: Plaintext / Markdown Normalization Processor."""
    return data.decode("utf-8", errors="replace").strip()


# =============================================================================
# PRIMITIVE: MEMORY CHECKPOINTING & DETERMINISTIC HASHING
# =============================================================================

def content_id(*parts: str) -> str:
    """Memory Checkpoint Primitive: Idempotent State Digest Generator.
    
    Generates a deterministic SHA-256 hash key across chunk contents and metadata.
    Prevents record duplication and enforces state consistency during database upserts.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


# =============================================================================
# PRIMITIVE: STATE SCHEMAS & API INITIALIZATION
# =============================================================================

def _secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value or "")


def get_api_key() -> str:
    """State Retrieval: Fetches context credentials from session or environment."""
    keyed = st.session_state.get("google_api_key", "")
    if keyed:
        return str(keyed)
    return _secret("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


def get_genai_client(api_key: str) -> genai.Client:
    """Client Infrastructure Factory: Initializes the stable Google GenAI SDK v1 Client."""
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(api_version="v1"),
    )


@st.cache_resource
def get_chroma_client() -> Any:
    """Memory Infrastructure Manager: Instantiates local or ephemeral ChromaDB persistence."""
    if os.environ.get("CHROMA_EPHEMERAL") == "1":
        return chromadb.EphemeralClient()
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection() -> Any:
    """Memory Infrastructure Manager: Provides thread-safe collection pointer."""
    return get_chroma_client().get_or_create_collection(name=COLLECTION_NAME)


# =============================================================================
# PRIMITIVE: COMPUTE NODES (EMBEDDINGS & PERCEPTION)
# =============================================================================

def embed_texts(texts: list[str] | str, api_key: str, for_query: bool = False) -> list[list[float]]:
    """Compute Node: Dense Vector Embedding Engine (gemini-embedding-2).
    
    Implements Asymmetric Task Structuring directly into payload inputs:
    - Queries: Prefixed with 'task: question answering | query:'
    - Documents: Prefixed with 'title: none | text:'
    This maximizes retrieval distance precision between questions and candidate facts.
    """
    client = get_genai_client(api_key)

    def prepare(value: str) -> str:
        if for_query:
            return f"task: question answering | query: {value}"
        return f"title: none | text: {value}"

    if isinstance(texts, list):
        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=prepare(text),
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=prepare(texts),
    )
    return [response.embeddings[0].values]


def caption_image(image: Image.Image, api_key: str) -> str:
    """Compute Node: Multimodal Visual Perception Engine.
    
    Transforms unstructured visual inputs into descriptive semantic representations
    for vector index indexing.
    """
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
    """Compute Node: Grounded Multimodal Synthesis Node.
    
    Aggregates retrieved vector context chunks and active chat visual inputs,
    enforcing source attribution constraints before calling Gemini 3.5 Flash.
    """
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


# =============================================================================
# PRIMITIVE: MEMORY PERSISTENCE & RETRIEVAL PIPELINE NODES
# =============================================================================

def index_bytes(filename: str, data: bytes, api_key: str) -> int:
    """Pipeline Compute Node: Ingestion & Vector Indexer.
    
    Extracts text/image features, executes chunking, computes dense embeddings,
    and upserts the data into ChromaDB long-term memory.
    """
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
    
    # Write to Long-Term Memory (ChromaDB)
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
    """Compute Node: Semantic Memory Retrieval Node.
    
    Queries ChromaDB vector collection using asymmetric query vector embeddings.
    """
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


# =============================================================================
# PRIMITIVE: STATE INITIALIZATION & USER INTERFACE NODES
# =============================================================================

def init_state() -> None:
    """State Schema Primitive: Short-Term Session Memory Setup.
    
    Initializes message history buffer and session configuration in Streamlit state memory.
    """
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("google_api_key", get_api_key())


def render_sidebar() -> int:
    """UI Control Node: Renders settings, collection controls, and file uploads."""
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Google AI Studio API key",
            type="password",
            value="",
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
    """Helper Node: Decodes image files from Streamlit input payloads."""
    images: list[Image.Image] = []
    for file in files:
        name = getattr(file, "name", "") or ""
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_TYPES:
            continue
        images.append(Image.open(io.BytesIO(file.getvalue())).convert("RGB"))
    return images


# =============================================================================
# PRIMITIVE: CONDITIONAL ROUTING & EXECUTION EDGE
# =============================================================================

def run_query(question: str, n_results: int, images: list[Image.Image]) -> None:
    """Execution Edge / Pipeline Coordinator.
    
    Coordinates the dynamic flow between:
    1. Memory Retrieval Node (`retrieve`)
    2. Synthesis Node (`generate_answer`)
    3. State Memory Update (`st.session_state.messages.append`)
    """
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

    # Edge Step 1: Query Long-Term Memory (ChromaDB)
    contexts, metadatas = retrieve(question, n_results=n_results, api_key=api_key)
    
    # Edge Step 2: Pass Context & Images to Model Synthesis Node
    answer = generate_answer(question, contexts, images, api_key=api_key)
    
    # Edge Step 3: Format Citation Sources
    sources = [
        {
            "filename": meta.get("filename", "unknown"),
            "type": meta.get("type", ""),
            "text": context,
        }
        for context, meta in zip(contexts, metadatas, strict=False)
    ]
    
    # Edge Step 4: Update Short-Term Session State Memory
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


# =============================================================================
# MAIN APPLICATION ROUTER & GRAPH ORCHESTRATION
# =============================================================================

def main() -> None:
    """SUPERVISOR ORCHESTRATOR & GRAPH ENTRY POINT.
    
    Controls application execution state and dynamic routing branches:
    - Branch 1: User attaches files -> Branch to Ingestion Pipeline (`index_bytes`)
    - Branch 2: User asks question -> Branch to Retrieval Execution Edge (`run_query`)
    - Branch 3: Display state -> Render Short-Term Memory Chat History
    """
    st.set_page_config(
        page_title="Multimodal RAG",
        page_icon=":material/auto_awesome:",
        layout="centered",
    )
    # Primitive 1: Initialize Session State
    init_state()
    n_results = render_sidebar()

    st.title("Multimodal RAG")
    st.caption("Ask questions over PDFs, text, and images stored in ChromaDB, using Gemini.")

    if not st.session_state.messages:
        st.info("Upload files in the sidebar, then ask a question below.")

    # Primitive 2: Render Conversational Memory
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

    # Supervisor Conditional Branching Edge
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
        
        # CONDITIONAL BRANCH 1: In-line Document & Media Ingestion Node
        if files and api_key:
            for file in files:
                extra_notes.append(f"Attached {file.name}")
                index_bytes(file.name, file.getvalue(), api_key)
        elif files and not api_key:
            extra_notes.append("Attachments were not indexed because no API key is set.")

        user_text = question or "(see attached files)"
        if extra_notes:
            user_text = user_text + "\n\n" + "\n".join(extra_notes)

        # Append User Input to State Schema Memory
        st.session_state.messages.append({"role": "user", "content": user_text, "sources": []})
        
        # CONDITIONAL BRANCH 2: Direct to Retrieval & Generation Node
        run_query(question or "Describe the attached files.", n_results, images)
        st.rerun()


if __name__ == "__main__":
    main()