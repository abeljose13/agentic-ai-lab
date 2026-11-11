"""
Test en vivo del buscador semántico (Clase 5) sobre el índice ChromaDB.

Carga los chunks + embeddings del laboratorio previo (JSON en /embeddings) a una
colección persistente, ejecuta 4 consultas de prueba (3 en dominio + 1 fuera de
dominio a propósito), verifica la persistencia del índice y demuestra el filtro
por metadata. Embebe las consultas con el proveedor de LLM_PROVIDER (llamadas
reales, mismo modelo que generó los vectores del corpus).
Ejecutar desde la raíz: uv run python tests/test_semantic_search.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz del repo para imports src.*

import chromadb

from src.llm_client.config import get_provider_name
from src.search.chroma_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    build_or_open_collection,
    collection_embedding_model,
    latest_run_path,
)
from src.search.searcher import MODEL_THRESHOLDS, search

TEST_QUERIES = [
    # (consulta, document_id esperado del chunk más relevante)
    ("¿Cuáles son las etapas de la ingesta de un sistema RAG?", "sistemas-rag-fundamentos"),
    ("¿Qué papel juega la clorofila en la fotosíntesis?", "la-fotosintesis-en-las-plantas"),
    ("¿Qué errores arruinan un lote de café durante el tueste?", "cafe-de-especialidad-en-argentina"),
    # Fuera de dominio, a propósito: debe devolver pocos o ningún resultado gracias al umbral.
    ("¿Cómo se hace un backup de la base de datos?", None),
]


def print_hits(query: str, hits: list[dict]) -> None:
    print(f"Q: {query}")
    if not hits:
        print("  -> (no results above the threshold)")
    for h in hits:
        meta = h["metadata"]
        print(f"  {h['chunk_id']}  (sim={h['similarity']:.3f})  "
              f"[document_id={meta['document_id']} position={meta['position']}]")
    print()


def main() -> int:
    # Umbral: se resuelve por modelo del índice (MODEL_THRESHOLDS, calibrado con
    # tests/calibrate_threshold.py); no hay un umbral global válido para todos los modelos.
    run_path = latest_run_path()
    collection = build_or_open_collection(run_path)
    stored_model = collection_embedding_model(collection)
    threshold = MODEL_THRESHOLDS[stored_model]
    print(f"=== SEMANTIC SEARCH (provider: {get_provider_name()}, model: {stored_model}, "
          f"threshold: {threshold}) ===\n")

    # Paso 1: cargar el run previo a la colección persistente (sin re-embeber nada).
    total = collection.count()
    print(f"Index built from {run_path.name} -> collection '{COLLECTION_NAME}': {total} chunks\n")
    assert total > 0, "The collection is empty"

    # Paso 2: consultas de prueba contra el índice real (evaluación manual de resultados).
    for query, expected_document_id in TEST_QUERIES:
        hits = search(query, collection, k=3, similarity_threshold=threshold)
        print_hits(query, hits)
        if expected_document_id is not None:
            # Criterio 1: consulta relevante -> chunks relevantes con chunk_id y metadata.
            assert hits, f"No results for in-domain query: '{query}'"
            # El documento correcto debe aparecer entre los resultados (no necesariamente en top-1).
            assert any(h["metadata"]["document_id"] == expected_document_id for h in hits), (
                f"No chunk from '{expected_document_id}' in the results for '{query}'"
            )
        else:
            # Criterio 2: consulta fuera de dominio -> pocos o ningún resultado (no 3 forzados).
            print(f"  -> out-of-domain query returned {len(hits)} results above threshold\n")

    # Paso 3: persistencia del índice — reabrir la colección sin re-embeber nada.
    new_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    reopened_collection = new_client.get_collection(COLLECTION_NAME)
    print(f"Reopened collection '{COLLECTION_NAME}': {reopened_collection.count()} chunks "
          f"(nothing re-embedded)")
    assert reopened_collection.count() == total, "Chunk count changed after reopening"

    # Paso 4: filtro por metadata (where) — buscar solo dentro de un documento.
    query = "¿Dónde se fija el dióxido de carbono durante la fotosíntesis?"
    hits = search(query, reopened_collection, k=3, similarity_threshold=threshold,
                  where={"document_id": "la-fotosintesis-en-las-plantas"})
    print(f"\nFiltered query (where document_id='la-fotosintesis-en-las-plantas'):")
    print_hits(query, hits)
    assert hits, "Filtered query returned no results"
    assert all(h["metadata"]["document_id"] == "la-fotosintesis-en-las-plantas" for h in hits), \
        "where filter returned chunks from another document"

    print("PASS  in-domain queries hit the right document, out-of-domain filtered by "
          "threshold, index persists and metadata filter works")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"\nFAIL  {type(err).__name__}: {err}")
        sys.exit(1)
