from core.data_store import get_s3_object_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import numpy as np
import re
import unicodedata
import json
import hashlib

def get_s3_etag_hash(filename: str) -> str:
    """
    Returns the S3 ETag used as a fast identity hash for cache lookup.
    Note: ETag is a quick change signal, not a guaranteed content SHA-256.
    """
    metadata = get_s3_object_metadata(filename)
    etag = metadata.get("etag")

    if not etag:
        raise ValueError(f"Couldn't get ETag for S3 object: {filename}")

    return etag

def _normalize_doc_name(doc_name: str) -> str:
    # 1) quita extensión .md si viene
    base = Path(doc_name).stem

    # 2) elimina tildes/acentos: Política -> Politica
    no_accents = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")

    # 3) deja solo letras, numeros, guion y underscore
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", no_accents).strip("_")

    return safe or "document"
    
def get_cache_paths(doc_name: str, file_hash: str) -> tuple[Path, Path]:
    """
    Returns cache paths under core/assets/embeddings:
    - <safe_doc_name>-<hash>.npy
    - <safe_doc_name>-<hash>.json
    """
    safe_doc_name = _normalize_doc_name(doc_name)
    safe_hash = re.sub(r"[^A-Za-z0-9_-]+", "", file_hash)

    core_dir = Path(__file__).resolve().parents[1]
    cache_dir = core_dir / "assets" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{safe_doc_name}-{safe_hash}"
    npy_path = cache_dir / f"{base_name}.npy"
    json_path = cache_dir / f"{base_name}.json"

    return npy_path, json_path

    
def cache_is_valid(doc_name: str, s3_hash: str) -> bool:
    """
    Returns True only if cache files exist and metadata matches the expected hash.
    """
    if not s3_hash:
        return False
    npy_path, json_path = get_cache_paths(doc_name, s3_hash)

    if not npy_path.exists() or not json_path.exists():
        return False

    try:
        with json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    cached_hash = metadata.get("file_hash")
    cached_content_sha = metadata.get("content_sha256")
    chunks = metadata.get("chunks")

    if cached_hash != s3_hash:
        return False

    # Strong identity must match too when available.
    if cached_content_sha and cached_content_sha != s3_hash:
        return False

    if not isinstance(chunks, list):
        return False

    return True
    

def load_cached_embeddings(doc_name: str, file_hash: str) -> tuple[np.ndarray, list[dict]]:
    """
    Loads cached embeddings (.npy) and chunk metadata (.json) for a document/hash pair.
    Raises ValueError if cache files are missing or malformed.
    """
    npy_path, json_path = get_cache_paths(doc_name, file_hash)

    if not npy_path.exists() or not json_path.exists():
        raise ValueError(
            f"Cache files not found for doc={doc_name}, hash={file_hash}"
        )

    try:
        embeddings = np.load(npy_path)
    except Exception as exc:
        raise ValueError(f"Invalid embeddings file: {npy_path}") from exc

    try:
        with json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid metadata file: {json_path}") from exc

    chunks = metadata.get("chunks")
    cached_hash = metadata.get("file_hash")

    if cached_hash != file_hash:
        raise ValueError(
            f"Hash mismatch in metadata. expected={file_hash}, got={cached_hash}"
        )

    if not isinstance(chunks, list):
        raise ValueError("Metadata field 'chunks' must be a list.")

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings array must be 2D. got shape={embeddings.shape}")

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Embeddings/chunks length mismatch: "
            f"{embeddings.shape[0]} vectors vs {len(chunks)} chunks"
        )

    return embeddings.astype(np.float32), chunks
    

def save_embeddings_cache(
    doc_name: str,
    file_hash: str,
    embeddings: np.ndarray,
    metadata: list[dict],
    s3_etag: str | None = None,
    content_sha256: str | None = None,
) -> None:
    """
    Saves embeddings (.npy) and metadata (.json) for a document/hash pair.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D. got shape={embeddings.shape}")

    if not isinstance(metadata, list):
        raise ValueError("metadata must be a list of chunk dictionaries.")

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embeddings/metadata length mismatch: "
            f"{embeddings.shape[0]} vectors vs {len(metadata)} chunks"
        )

    npy_path, json_path = get_cache_paths(doc_name, file_hash)

    # 1) Save vectors
    np.save(npy_path, embeddings.astype(np.float32))

    # 2) Save metadata envelope
    payload = {
        "doc_name": doc_name,
        "file_hash": file_hash,
        "s3_etag": s3_etag,
        "content_sha256": content_sha256 or file_hash,
        "chunk_count": len(metadata),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunks": metadata,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def compute_content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_or_compute_embeddings(
    doc_name: str,
    md_text: str,
    chunk_fn: Callable[[str, str], list[dict]],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    """
    Robust cache orchestration.
    Uses content SHA-256 as source of truth and stores ETag as auxiliary metadata.
    """
    if not isinstance(md_text, str) or not md_text.strip():
        raise ValueError(f"Document is empty or invalid: {doc_name}")

    # Fast identity signal from S3 metadata (not authoritative for content).
    s3_etag = get_s3_etag_hash(doc_name)

    # Strong identity of document content.
    content_sha = compute_content_sha256(md_text)

    # 100% safe reuse: only when content hash matches the cache identity.
    if cache_is_valid(doc_name, content_sha):
        embeddings, chunks = load_cached_embeddings(doc_name, content_sha)
        return embeddings, chunks

    chunks = chunk_fn(md_text, doc_name)
    if not chunks:
        return np.empty((0, 0), dtype=np.float32), []

    chunk_texts = [c["embed_text"] for c in chunks]
    embeddings = embed_fn(chunk_texts)

    save_embeddings_cache(
        doc_name=doc_name,
        file_hash=content_sha,
        embeddings=embeddings,
        metadata=chunks,
        s3_etag=s3_etag,
        content_sha256=content_sha,
    )

    return embeddings.astype(np.float32), chunks