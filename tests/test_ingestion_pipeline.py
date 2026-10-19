"""
Test en vivo del pipeline de ingesta sobre el corpus completo.
Usa el proveedor de embeddings configurado en LLM_PROVIDER (hace llamadas reales).
Ejecutar desde la raíz: uv run python tests/test_ingestion_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz del repo para imports src.*

import numpy as np

from src.ingestion.corpus import EXAMPLE_CORPUS, load_corpus
from src.ingestion.pipeline import run_ingestion
from src.ingestion.storage import find_chunk
from src.llm_client.config import get_provider_name

CHUNK_LIMIT = 1500


def main() -> int:
    print(f"=== LIVE INGESTION PIPELINE (provider: {get_provider_name()}) ===\n")
    documents = load_corpus(EXAMPLE_CORPUS)
    print(f"Corpus: {len(documents)} documents from {EXAMPLE_CORPUS.name}")

    # Criterio 1: el pipeline sobre el corpus completo no tira errores
    result = run_ingestion(documents)
    chunks, vectors, path = result["chunks"], result["vectors"], result["path"]
    print(f"Chunks generated: {len(chunks)} | vectors: {len(vectors)} | file: {path}")
    assert len(chunks) == len(vectors) > 0

    dimensions = {len(v) for v in vectors}
    assert len(dimensions) == 1, f"Inconsistent dimensions: {dimensions}"
    print(f"Embedding dimension: {dimensions.pop()}")

    # Criterio 2: cada chunk < ~1500 caracteres
    longest = max(len(c.text) for c in chunks)
    assert longest < CHUNK_LIMIT, f"Chunk with {longest} characters"
    print(f"Longest chunk: {longest} characters (limit {CHUNK_LIMIT})")

    # Criterio 3: cualquier chunk_id -> documento y posición sin ambigüedad (desde el archivo guardado)
    for c in chunks:
        ref = find_chunk(path, c.chunk_id)
        assert ref["document_id"] == c.document_id and ref["position"] == c.position, c.chunk_id
    print(f"Traceability verified for {len(chunks)} chunk_id values")

    # Sanity check semántico: chunks del mismo documento deben parecerse más que de documentos distintos
    m = np.array(vectors)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    same = [sim[i, j] for i in range(len(chunks)) for j in range(i + 1, len(chunks))
            if chunks[i].document_id == chunks[j].document_id]
    other = [sim[i, j] for i in range(len(chunks)) for j in range(i + 1, len(chunks))
             if chunks[i].document_id != chunks[j].document_id]
    if same and other:
        print(f"Average cosine similarity: same document {np.mean(same):.3f} | different document {np.mean(other):.3f}")

    print("\nPASS  full pipeline ran without errors, chunks < 1500, traceability by chunk_id")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"\nFAIL  {type(err).__name__}: {err}")
        sys.exit(1)
