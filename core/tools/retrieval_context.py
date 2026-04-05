from core.data_store import load_s3_data
from dotenv import load_dotenv
from strands import tool
import numpy as np
import boto3
import json
import os
import time

load_dotenv()

client = boto3.client(
    'bedrock-runtime', 
    region_name=os.getenv("AWS_REGION", "us-east-2")
    )

# Cache: embed_text -> embedding vector. Avoids re-embedding same chunks across calls.
_EMBED_CACHE: dict[str, list[float]] = {}


def chunk_md_by_sections(md_text: str, doc_name: str) -> list[dict]:
    """
    Splits markdown into chunks by section header (##).
    doc_name can be passed as a filename, e.g. "Políticas de envío.md".
    Returns a list of dicts with keys: embed_text, doc_name, section_title, content.
    """
    clean_doc_name = os.path.splitext(os.path.basename(doc_name.strip()))[0]

    chunks: list[dict] = []
    current_section: str | None = None
    current_lines: list[str] = []

    for line in md_text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if current_section is not None:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({
                        "embed_text": (
                            f"Documento: {clean_doc_name}\n"
                            f"Sección: {current_section}\n\n"
                            f"Contenido:\n{content}"
                        ),
                        "doc_name": clean_doc_name,
                        "section_title": current_section,
                        "content": content,
                    })
            current_section = line[3:].strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section is not None:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({
                "embed_text": (
                    f"Documento: {clean_doc_name}\n"
                    f"Sección: {current_section}\n\n"
                    f"Contenido:\n{content}"
                ),
                "doc_name": clean_doc_name,
                "section_title": current_section,
                "content": content,
            })

    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embeds a list of strings using Titan Embed v2.
    Uses module-level cache to avoid re-embedding identical texts.
    Returns a 2D numpy array of shape (len(texts), embedding_dim).
    """
    embeddings = []
    for text in texts:
        if text not in _EMBED_CACHE:
            response = client.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": text})
            )
            _EMBED_CACHE[text] = json.loads(response['body'].read())["embedding"]
        embeddings.append(_EMBED_CACHE[text])
    return np.array(embeddings, dtype=np.float32)


def cosine_similarity_batch(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    # Norm
    query = query_vec.astype(np.float32)
    docs = doc_vecs.astype(np.float32)

    query = query / np.linalg.norm(query)
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    # Dot product = CS when normalized!
    return docs @ query  # shape: (num_docs,)


@tool
def retrieval_context(query: str) -> list[dict]:
    """
    Retrieves the most relevant policy sections for a given query
    using Bedrock embeddings and cosine similarity.
    Args:
      query: Natural language question or user message.
    Returns:
      List of up to 3 dicts, each with keys:
        doc_name, section_title, breadcrumb, content.
    """
    # 1. Load policy docs from S3 (cached after first call)
    s = time.perf_counter()
    docs = {
        "Políticas de envío.md":       load_s3_data("Políticas de envío.md"),
        "Política de garantía.md":     load_s3_data("Política de garantía.md"),
        "Política de devoluciones.md": load_s3_data("Política de devoluciones.md"),
    }
    f = time.perf_counter()
    print(f"Load all s3 files: {s - f}")

    # 2. Chunk all docs by ## sections
    s = time.perf_counter()

    all_chunks: list[dict] = []
    for key, md_text in docs.items():
        if isinstance(md_text, str):
            all_chunks.extend(chunk_md_by_sections(md_text, key))

    if not all_chunks:
        return []
    f = time.perf_counter()
    print(f"Chunk all files: {s - f}")

    s = time.perf_counter()
    # 3. Embed all chunks (cached) and the query
    chunk_texts = [c["embed_text"] for c in all_chunks]
    chunk_vecs = embed_texts(chunk_texts)          # (n_chunks, dim)
    query_vec  = embed_texts([query])[0]           # (dim,)

    f = time.perf_counter()
    print(f"Embed all chunks and query: {s - f}")

    s = time.perf_counter()
    # 4. Cosine similarity and top-3
    scores = cosine_similarity_batch(query_vec, chunk_vecs)
    top_k = min(3, len(all_chunks))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    f = time.perf_counter()
    print(f"Cosine sim: {s - f}")

    # 5. Build result
    results = []
    for idx in top_indices:
        chunk = all_chunks[idx]
        results.append({
            "doc_name":      chunk["doc_name"],
            "section_title": chunk["section_title"],
            "breadcrumb":    f"{chunk['doc_name']} > {chunk['section_title']}",
            "content":       chunk["content"],
        })

    return results

if __name__  == "__main__":
    results = retrieval_context("Que cubre la garantia?")
    for ch in results:
        print(ch["section_title"])


