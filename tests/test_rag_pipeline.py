"""
Validación manual del pipeline RAG integrado (Clase 6).

Ejecuta 4 preguntas reales sobre el corpus indexado, imprime respuestas y verifica
mecánicamente que el contexto no desborde el presupuesto y que las citas resuelvan
a chunks reales. Hace llamadas reales de embeddings y LLM según LLM_PROVIDER.

Ejecutar desde la raíz: uv run python tests/test_rag_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_client.config import get_provider_name
from src.search.chroma_store import build_or_open_collection, collection_embedding_model, latest_run_path
from src.search.searcher import MODEL_THRESHOLDS
from src.workflows.rag import NO_EVIDENCE_MESSAGE, rag

# Cuatro preguntas de prueba: 2 en dominio, 1 ambigua, 1 fuera de dominio.
TEST_QUESTIONS = [
    # Buena respuesta esperada: RAG.
    "¿Cuáles son las etapas de la ingesta de un sistema RAG?",
    # Buena respuesta esperada: fotosíntesis.
    "¿Qué papel juega la clorofila en la fotosíntesis?",
    # Ambigua: "temperatura" aparece en fotosíntesis y en preparación de café.
    "¿Qué papel juega la temperatura?",
    # Fuera de dominio: no está en el corpus.
    "¿Cómo se hace un backup de la base de datos?",
]


def check_context_budget(result, budget: int) -> None:
    assert result.context_characters <= budget, (
        f"Context length {result.context_characters} exceeds budget {budget}"
    )


def check_sources_are_real(result, collection) -> None:
    for source in result.sources:
        chunk_id = source["chunk_id"]
        fetched = collection.get(ids=[chunk_id], include=["metadatas"])
        assert fetched["ids"] and fetched["ids"][0] == chunk_id, (
            f"Citation points to non-existent chunk_id: {chunk_id}"
        )


def print_result(i: int, question: str, result) -> None:
    print(f"--- Question {i}: {question} ---")
    print(f"Response: {result.response}")
    print(f"Threshold: {result.threshold:.3f} | Raw hits: {len(result.raw_hits)} | "
          f"Filtered hits: {len(result.filtered_hits)} | Context hits: {len(result.context_hits)}")
    print("Raw hits:")
    for h in result.raw_hits:
        marker = "PASS" if h["similarity"] >= result.threshold else "FAIL"
        print(f"  [{marker}] {h['chunk_id']} sim={h['similarity']:.3f}")
    print("Context hits:")
    for h in result.context_hits:
        print(f"  [{h['chunk_id']}] sim={h['similarity']:.3f} chars={len(h['text'])}")
    if result.discarded_by_budget:
        print(f"Discarded by budget: {result.discarded_by_budget}")
    print("Sources cited:")
    for s in result.sources:
        print(f"  [{s['ref']}] {s['chunk_id']} (doc={s['document_id']}, pos={s['position']})")
    print()


def main() -> int:
    run_path = latest_run_path()
    collection = build_or_open_collection(run_path)
    stored_model = collection_embedding_model(collection)
    threshold = MODEL_THRESHOLDS[stored_model]
    budget = 1200  # presupuesto deliberadamente bajo para forzar truncamiento si hay chunks largos

    print(f"=== RAG PIPELINE TEST (provider: {get_provider_name()}, model: {stored_model}, "
          f"threshold: {threshold}, context_budget: {budget}) ===\n")
    print(f"Collection: {collection.count()} chunks from {run_path.name}\n")

    passed = 0
    for i, question in enumerate(TEST_QUESTIONS, start=1):
        result = rag(
            query=question,
            collection=collection,
            k=3,
            similarity_threshold=threshold,
            context_budget_chars=budget,
            temperature=0.3,
            max_tokens=400,
        )
        print_result(i, question, result)

        check_context_budget(result, budget)
        check_sources_are_real(result, collection)

        # Criterios específicos por tipo de pregunta.
        if i == 4:
            # Fuera de dominio: debe abstenerse.
            assert result.no_evidence, "Out-of-domain question should produce no_evidence"
            assert result.response == NO_EVIDENCE_MESSAGE, "Out-of-domain response must be the no-evidence message"
            assert not result.sources, "Out-of-domain question should not cite sources"
        else:
            # En dominio / ambigua: respuesta sustantiva con al menos una cita real.
            assert not result.no_evidence, f"Expected an answer for question {i}"
            assert result.sources, f"Question {i} must cite at least one real chunk"

        passed += 1

    print(f"PASS  all {passed}/{len(TEST_QUESTIONS)} questions validated")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as err:
        print(f"\nFAIL  {type(err).__name__}: {err}")
        sys.exit(1)
