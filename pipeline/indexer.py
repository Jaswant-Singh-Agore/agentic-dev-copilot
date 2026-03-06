"""
Handles codebase indexing and semantic search using FAISS.
Chunks Python files, generates embeddings via HuggingFace, and builds a searchable index.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from huggingface_hub import InferenceClient

from dev_copilot_config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    HF_API_TOKEN,
    SAMPLE_CODE_PATH,
    TOP_K_SIMILAR,
)


logger = logging.getLogger(__name__)

_INDEX_FILE = "code.index"
_CHUNKS_FILE = "chunks.pkl"

# single client instance reused across all embedding calls
_client = InferenceClient(api_key=HF_API_TOKEN)


def get_embedding(text: str) -> np.ndarray | None:
    """
    Generate a normalized embedding vector for a text snippet.
    Returns None if the API call fails.
    """
    try:
        response = _client.feature_extraction(text, model=EMBEDDING_MODEL)
        vec = np.array(response, dtype=np.float32)

        if vec.ndim == 2:
            vec = vec.mean(axis=0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


def chunk_code(code: str, file_path: str, chunk_size: int = 50) -> list[dict]:
    """
    Split a Python file into overlapping chunks of roughly chunk_size lines.
    Overlap is chunk_size // 2 to preserve context across boundaries.
    """
    lines = code.splitlines()
    chunks = []
    step = chunk_size // 2

    for i in range(0, max(1, len(lines) - chunk_size + 1), step):
        chunk_lines = lines[i:i + chunk_size]
        chunk_text = "\n".join(chunk_lines)
        if chunk_text.strip():
            chunks.append({
                "content": chunk_text,
                "file": file_path,
                "start_line": i + 1,
                "end_line": i + len(chunk_lines),
            })

    # make sure the final lines are always covered
    if lines and len(lines) > chunk_size:
        last_chunk = "\n".join(lines[-chunk_size:])
        if last_chunk.strip():
            chunks.append({
                "content": last_chunk,
                "file": file_path,
                "start_line": len(lines) - chunk_size + 1,
                "end_line": len(lines),
            })

    return chunks


def build_index(code_dir: str | None = None) -> dict:
    """
    Build a FAISS index from all Python files in code_dir.
    Saves the index and chunk metadata to FAISS_INDEX_PATH.
    """
    code_dir = code_dir or SAMPLE_CODE_PATH
    index_dir = Path(FAISS_INDEX_PATH)
    index_dir.mkdir(parents=True, exist_ok=True)

    py_files = list(Path(code_dir).rglob("*.py"))
    if not py_files:
        logger.warning("No .py files found in '%s'.", code_dir)
        return {
            "status": "warning",
            "message": f"No .py files found in {code_dir}",
            "files_indexed": 0,
            "chunks_indexed": 0,
        }

    all_chunks: list[dict] = []
    all_embeddings: list[np.ndarray] = []

    logger.info("Indexing %d Python file(s) from '%s'...", len(py_files), code_dir)

    for py_file in py_files:
        try:
            code = py_file.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_code(code, str(py_file))
            for chunk in chunks:
                emb = get_embedding(chunk["content"])
                if emb is not None:
                    all_chunks.append(chunk)
                    all_embeddings.append(emb)
            logger.info("  %s — %d chunk(s)", py_file.name, len(chunks))
        except Exception as e:
            logger.warning("Skipping '%s': %s", py_file.name, e)

    if not all_embeddings:
        return {
            "status": "error",
            "message": "No embeddings generated.",
            "files_indexed": 0,
            "chunks_indexed": 0,
        }

    dim = len(all_embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.vstack(all_embeddings).astype(np.float32))

    faiss.write_index(index, str(index_dir / _INDEX_FILE))
    with open(index_dir / _CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info("Index saved to '%s' — %d chunk(s).", index_dir, len(all_chunks))

    return {
        "status": "success",
        "files_indexed": len(py_files),
        "chunks_indexed": len(all_chunks),
        "index_path": str(index_dir),
        "embedding_dim": dim,
    }


def load_index() -> tuple[faiss.IndexFlatIP | None, list[dict]]:
    """Load a saved FAISS index and chunk metadata from disk."""
    index_dir = Path(FAISS_INDEX_PATH)
    index_file = index_dir / _INDEX_FILE
    chunks_file = index_dir / _CHUNKS_FILE

    if not index_file.exists() or not chunks_file.exists():
        return None, []

    index = faiss.read_index(str(index_file))
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search_codebase(query: str, top_k: int | None = None) -> list[dict]:
    """
    Search the indexed codebase for chunks semantically similar to the query.
    Returns an empty list if the index is not built or the embedding fails.
    """
    top_k = top_k or TOP_K_SIMILAR
    index, chunks = load_index()

    if index is None or index.ntotal == 0:
        return []

    query_emb = get_embedding(query)
    if query_emb is None:
        return []

    scores, indices = index.search(
        query_emb.reshape(1, -1).astype(np.float32),
        min(top_k, index.ntotal),
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        results.append(chunk)

    return results


def reset_index() -> bool:
    """Delete saved index files. Returns True if any files were removed."""
    index_dir = Path(FAISS_INDEX_PATH)
    removed = False
    for fname in [_INDEX_FILE, _CHUNKS_FILE]:
        fpath = index_dir / fname
        if fpath.exists():
            fpath.unlink()
            removed = True
    return removed