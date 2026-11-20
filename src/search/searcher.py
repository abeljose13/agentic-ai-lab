"""
Buscador semántico sobre la colección ChromaDB del laboratorio.

Dado un texto de consulta, embebe la consulta con el mismo proveedor/modelo
que generó los vectores del corpus y devuelve el top-k de la colección,
descartando los resultados por debajo de un umbral de similitud mínimo.
"""
from src.ingestion.embeddings import get_embed_fn, get_embed_model
from src.search.chroma_store import collection_embedding_model

# Umbral de similitud calibrado por modelo de embeddings (ver tests/calibrate_threshold.py).
# Cada modelo tiene su propia escala de similitudes, por eso no hay un umbral global.
MODEL_THRESHOLDS: dict[str, float] = {
    "mxbai-embed-large": 0.62,  # calibración previa: los rangos se solapan (backup BD ~0.67 vs RAG ~0.65)
    # Calibración 2026-11-02 (5 consultas en dominio + 4 fuera): separación limpia
    # (en dominio 0.25-0.46 vs fuera de dominio <=0.19), corte en el punto medio.
    "nvidia/llama-nemotron-embed-vl-1b-v2:free": 0.22,
    # Calibración 2026-11-20 (8 consultas en dominio + 4 fuera): separación limpia
    # (worst in-domain 0.464 vs best out-of-domain 0.458), corte en el punto medio.
    "bge-m3": 0.461,
    # Calibración 2026-11-16 con gemini-embedding-2 (5 en dominio + 4 fuera de dominio).
    # worst in-domain top-1 = 0.637, best out-of-domain top-1 = 0.546 -> corte en el punto medio.
    "gemini-embedding-2": 0.592,
}


def _guard_embedding_model(collection) -> str:
    """Verifica que el modelo configurado coincida con el de la colección y lo devuelve."""
    stored_model = collection_embedding_model(collection)
    configured_model = get_embed_model()
    if stored_model != configured_model:
        raise ValueError(
            f"Corpus was embedded with '{stored_model}' but the configured model is "
            f"'{configured_model}'. Re-run ingestion with the configured model (the collection "
            f"is rebuilt automatically) or set the embedding model back to '{stored_model}'."
        )
    return stored_model


def _resolve_threshold(stored_model: str, similarity_threshold: float | None) -> float:
    """Resuelve el umbral efectivo: el explícito o el calibrado para el modelo del índice."""
    if similarity_threshold is not None:
        return similarity_threshold
    if stored_model not in MODEL_THRESHOLDS:
        raise ValueError(
            f"No calibrated similarity threshold for model '{stored_model}'. Run "
            f"tests/calibrate_threshold.py and add it to MODEL_THRESHOLDS, or pass "
            f"similarity_threshold explicitly."
        )
    return MODEL_THRESHOLDS[stored_model]


def search_with_details(query: str, collection, k: int = 3,
                        similarity_threshold: float | None = None,
                        where: dict = None) -> dict:
    """Busca los k chunks más relevantes y expone el resultado completo.

    Devuelve un dict con:
      - "hits": lista filtrada por encima del umbral (misma forma que `search()`).
      - "raw_hits": los k resultados brutos de ChromaDB, incluidos los que quedan
        por debajo del umbral.
      - "threshold": umbral efectivo aplicado.
    """
    stored_model = _guard_embedding_model(collection)
    threshold = _resolve_threshold(stored_model, similarity_threshold)

    # Embeber la consulta con input_type="search_query". Los proveedores que no distinguen
    # el tipo (Ollama, OpenAI) lo ignoran; Google lo mapea a RETRIEVAL_QUERY.
    embed_fn = get_embed_fn()
    query_vector = embed_fn([query], input_type="search_query")[0]

    # Consultar la colección (n_results=k, where=where).
    result = collection.query(query_embeddings=[query_vector], n_results=k, where=where)

    # La colección usa métrica coseno: distancia = 1 - similitud (más alto = más parecido).
    raw_hits = []
    hits = []
    for chunk_id, text, distance, metadata in zip(
        result["ids"][0],
        result["documents"][0],
        result["distances"][0],
        result["metadatas"][0],
    ):
        similarity = 1.0 - distance
        hit = {"chunk_id": chunk_id, "text": text, "similarity": similarity, "metadata": metadata}
        raw_hits.append(hit)
        if similarity >= threshold:
            hits.append(hit)

    return {"hits": hits, "raw_hits": raw_hits, "threshold": threshold}


def search(query: str, collection, k: int = 3, similarity_threshold: float | None = None,
           where: dict = None) -> list[dict]:
    """Busca los k chunks más relevantes para `query` en `collection`, descartando los
    que estén por debajo de `similarity_threshold`. `where` permite filtrar por metadata.

    Si `similarity_threshold` es None, se usa el umbral calibrado para el modelo con
    el que se construyó la colección (MODEL_THRESHOLDS). Un modelo sin calibrar lanza
    error: hay que calibrarlo antes de buscar.

    Devuelve una lista de dicts (ordenados de más a menos similar) con al menos
    {"chunk_id", "text", "similarity", "metadata"}.
    """
    return search_with_details(query, collection, k=k, similarity_threshold=similarity_threshold,
                               where=where)["hits"]
