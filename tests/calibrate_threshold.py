"""
Calibración del umbral de similitud para el buscador semántico (Clase 5).

Corre un benchmark de consultas (3 en dominio + 4 fuera de dominio) contra el
índice vigente sin umbral, mide la similitud del top-1 de cada consulta y sugiere
un corte. El valor elegido se registra en MODEL_THRESHOLDS (src/search/searcher.py),
que search() usa como default para ese modelo.

Hace llamadas reales de embeddings (una por consulta) con el proveedor de
LLM_PROVIDER. Ejecutar desde la raíz: uv run python tests/calibrate_threshold.py
"""
import sys
from pathlib import Path, PosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz del repo para imports src.*

from src.ingestion.embeddings import get_embed_model
from src.llm_client.config import get_provider_name
from src.search.chroma_store import build_or_open_collection, collection_embedding_model, latest_run_path
from src.search.searcher import MODEL_THRESHOLDS, search

# (consulta, document_id esperado) — deben recuperar un chunk del documento correcto.
IN_DOMAIN_QUERIES = [
    ("¿Cuáles son las etapas de la ingesta de un sistema RAG?", "sistemas-rag-fundamentos"),
    ("¿Qué papel juega la clorofila en la fotosíntesis?", "la-fotosintesis-en-las-plantas"),
    ("¿Qué errores arruinan un lote de café durante el tueste?", "cafe-de-especialidad-en-argentina"),
    ("¿Para qué sirve el floema en las plantas?", "la-fotosintesis-en-las-plantas"),
    ("¿Qué es el ciclo de Calvin?", "la-fotosintesis-en-las-plantas"),
]

# Fuera de dominio, a propósito: un umbral bien calibrado las filtra.
OUT_OF_DOMAIN_QUERIES = [
    "¿Cómo se hace un backup de la base de datos?",
    "¿Cuál es la receta de una tarta de manzana?",
    "¿Quién ganó la Copa del Mundo de fútbol 2022?",
    "¿Cómo cambio el aceite del motor de un auto?",
]


def main() -> int:
    model = get_embed_model()
    print(f"=== THRESHOLD CALIBRATION (provider: {get_provider_name()}, model: {model}) ===\n")

    # Índice vigente: la colección se reconstruye sola si cambió el modelo del run.
    run_path = latest_run_path()
    collection = build_or_open_collection(run_path)
    stored_model = collection_embedding_model(collection)
    print(f"Index from {run_path.name} -> {collection.count()} chunks (model: {stored_model})\n")
    assert stored_model == model, (
        f"Collection model '{stored_model}' != configured model '{model}': re-run ingestion first"
    )

    in_domain_top1 = []
    print("-- In-domain queries --")
    for query, expected_document_id in IN_DOMAIN_QUERIES:
        hits = search(query, collection, k=3, similarity_threshold=0.0)
        assert hits, f"No results at all for in-domain query: '{query}'"
        top1 = hits[0]
        ok = any(h["metadata"]["document_id"] == expected_document_id for h in hits)
        in_domain_top1.append(top1["similarity"])
        sims = ", ".join(f"{h['similarity']:.3f}" for h in hits)
        print(f"  {query}\n    top-3 sims: {sims}  | expected doc in results: {'YES' if ok else 'NO!'}")
    print()

    out_of_domain_top1 = []
    print("-- Out-of-domain queries --")
    for query in OUT_OF_DOMAIN_QUERIES:
        hits = search(query, collection, k=3, similarity_threshold=0.0)
        top1 = hits[0] if hits else None
        out_of_domain_top1.append(top1["similarity"] if top1 else 0.0)
        sims = ", ".join(f"{h['similarity']:.3f}" for h in hits) if hits else "(no results)"
        print(f"  {query}\n    top-3 sims: {sims}")
    print()

    # Corte sugerido: punto medio entre la peor consulta en dominio y la mejor fuera de dominio.
    worst_in = min(in_domain_top1)
    best_out = max(out_of_domain_top1)
    print(f"worst in-domain top-1:     {worst_in:.3f}")
    print(f"best out-of-domain top-1:  {best_out:.3f}")

    if best_out < worst_in:
        suggested = (worst_in + best_out) / 2
        print(f"separation: clean -> suggested threshold (midpoint): {suggested:.3f}")
    else:
        # Rangos solapados: no existe un corte que separe ambos casos. Conservador:
        # no perder ninguna consulta en dominio, aceptando que alguna fuera de dominio pase.
        suggested = worst_in
        leakers = [q for q, s in zip(OUT_OF_DOMAIN_QUERIES, out_of_domain_top1) if s >= suggested]
        print(f"separation: OVERLAP -> no clean cutoff exists")
        print(f"suggested threshold (keep all in-domain): {suggested:.3f}")
        print(f"out-of-domain queries that would still pass: {leakers}")

    if stored_model in MODEL_THRESHOLDS:
        print(f"\ncurrent registered threshold for '{stored_model}': {MODEL_THRESHOLDS[stored_model]}")
    print(f"\nNext: register the chosen value in MODEL_THRESHOLDS['{stored_model}'] "
          f"(src/search/searcher.py)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"\nFAIL  {type(err).__name__}: {err}")
        sys.exit(1)
