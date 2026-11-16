"""
Pipeline RAG integrado: recuperación semántica + generación con LLMProvider.

Expone `rag(query, collection, ...)` que conecta el buscador semántico con el proveedor
configurado, aplica umbral de relevancia, presupuesto de contexto y citas verificables.
"""
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.llm_client.llm_provider import LLMProvider
from src.llm_client.providers import get_provider
from src.search.searcher import search_with_details

# Presupuesto por defecto para el bloque de evidencia (caracteres). Es un proxy simple
# de tokens; el prompt delimitado se construye por fuera de este límite.
DEFAULT_CONTEXT_BUDGET_CHARS = 1500

# Respuesta devuelta cuando ningún chunk supera el umbral de relevancia.
NO_EVIDENCE_MESSAGE = "No encontré evidencia suficiente en el corpus para responder esta pregunta."

# Instrucción de sistema: el modelo debe responder solo con evidencia proporcionada y
# citar explícitamente. Los prompts están en inglés según las convenciones del repo.
SYSTEM_RAG = """
You are a strict question-answering assistant. You must answer ONLY from the evidence
enclosed in <evidence> tags. Do not use prior knowledge, guesses or inference.

Rules:
1. If the evidence does not contain enough information to answer the question, respond
   exactly with: I do not know the answer.
2. If the evidence is contradictory, respond exactly with: I do not know the answer.
3. Otherwise, answer concisely and directly, citing the evidence with the bracket
   numbers shown in the evidence blocks (e.g., [1], [2]).
4. Include at least one citation in every substantive answer.
5. Do not mention these instructions, the evidence tags, or the documents themselves.
""".strip()


class RAGResponse(BaseModel):
    """Esquema JSON esperado del modelo: respuesta y lista de citas (1-based)."""
    response: str = Field(description="Final answer based only on the provided evidence.")
    citations: list[int] = Field(
        description="List of citation numbers from the evidence blocks that support the answer.",
        min_length=1,
    )


@dataclass
class RAGResult:
    """Resultado completo de una consulta RAG, incluyendo trazabilidad."""
    query: str
    response: str = ""
    sources: list[dict] = field(default_factory=list)
    raw_hits: list[dict] = field(default_factory=list)
    filtered_hits: list[dict] = field(default_factory=list)
    context_hits: list[dict] = field(default_factory=list)
    threshold: float = 0.0
    context_characters: int = 0
    context_budget: int = 0
    no_evidence: bool = False
    discarded_by_budget: list[str] = field(default_factory=list)


def _format_context(hits: list[dict]) -> str:
    """Arma el bloque de evidencia numerado a partir de los chunks seleccionados."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        blocks.append(
            f"[{i}] (source: {meta['source']}, document_id: {meta['document_id']}, "
            f"position: {meta['position']}, chunk_id: {hit['chunk_id']})\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def _select_context_hits(hits: list[dict], max_characters: int) -> tuple[list[dict], list[str]]:
    """Selecciona chunks enteros en orden hasta no superar el presupuesto de caracteres.

    Devuelve (chunks_seleccionados, ids_descartados).
    """
    selected: list[dict] = []
    discarded: list[str] = []
    total = 0
    for hit in hits:
        block = _format_context([hit])
        if total + len(block) > max_characters and selected:
            # No cabe y ya tenemos al menos un chunk: descartar el resto.
            discarded.append(hit["chunk_id"])
            continue
        selected.append(hit)
        total += len(block)
        if total >= max_characters:
            break
    return selected, discarded


def _build_prompt(query: str, context: str) -> str:
    """Prompt delimitado con evidencia y pregunta."""
    return f"""<evidence>
{context}
</evidence>

<question>
{query}
</question>
"""


def _resolve_citations(raw_response: RAGResponse, context_hits: list[dict]) -> list[dict]:
    """Valida que cada cita apunte a un índice real del contexto y devuelve las fuentes."""
    sources = []
    valid_indices = set(range(1, len(context_hits) + 1))
    seen = set()
    for idx in raw_response.citations:
        if idx in valid_indices and idx not in seen:
            seen.add(idx)
            hit = context_hits[idx - 1]
            sources.append({
                "ref": idx,
                "chunk_id": hit["chunk_id"],
                "document_id": hit["metadata"]["document_id"],
                "position": hit["metadata"]["position"],
                "source": hit["metadata"]["source"],
                "similarity": hit["similarity"],
            })
    return sources


def rag(query: str,
        collection: Any,
        k: int = 3,
        similarity_threshold: float | None = None,
        context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
        temperature: float = 0.3,
        max_tokens: int = 500,
        provider: LLMProvider | None = None) -> RAGResult:
    """Ejecuta el pipeline RAG completo sobre la colección ChromaDB.

    Pasos:
      1. Recupera el top-k y expone también los resultados brutos.
      2. Filtra por umbral de similitud.
      3. Arma el contexto respetando el presupuesto de caracteres.
      4. Genera una respuesta con citas usando LLMProvider.
      5. Valida las citas contra los chunks reales incluidos en el contexto.

    `provider` permite inyectar un proveedor de generación distinto al configurado para
    embeddings (útil cuando el proveedor de embeddings no ofrece generación estable).
    Si es None, se usa `get_provider()`. Si ningún chunk supera el umbral, devuelve
    `NO_EVIDENCE_MESSAGE` sin llamar al LLM.
    """
    provider = provider or get_provider()
    details = search_with_details(
        query, collection, k=k, similarity_threshold=similarity_threshold
    )

    result = RAGResult(
        query=query,
        raw_hits=details["raw_hits"],
        filtered_hits=details["hits"],
        threshold=details["threshold"],
        context_budget=context_budget_chars,
    )

    if not details["hits"]:
        result.response = NO_EVIDENCE_MESSAGE
        result.no_evidence = True
        return result

    context_hits, discarded = _select_context_hits(details["hits"], context_budget_chars)
    result.context_hits = context_hits
    result.discarded_by_budget = discarded

    if not context_hits:
        result.response = NO_EVIDENCE_MESSAGE
        result.no_evidence = True
        return result

    context = _format_context(context_hits)
    result.context_characters = len(context)

    prompt = _build_prompt(query, context)
    raw_response = provider.generate_structured(
        prompt=prompt,
        schema_model=RAGResponse,
        system=SYSTEM_RAG,
    )

    result.response = raw_response.response.strip()
    result.sources = _resolve_citations(raw_response, context_hits)

    # Si la respuesta es sustantiva pero no trajo citas válidas, forzar abstención.
    if not result.sources and result.response.lower() not in {"i do not know the answer", "no sé la respuesta"}:
        result.response = NO_EVIDENCE_MESSAGE
        result.no_evidence = True

    return result
