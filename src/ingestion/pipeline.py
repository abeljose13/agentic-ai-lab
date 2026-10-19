from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.chunking import DEFAULT_MAX_CHARS, Chunk, build_chunks
from src.ingestion.cleaning import clean_text
from src.ingestion.corpus import Document
from src.ingestion.embeddings import EmbedFn, get_embed_fn
from src.ingestion.storage import save_embeddings
from src.llm_client.config import get_provider_name


def run_ingestion(documents: list[Document], embed_fn: EmbedFn | None = None,
                  max_chars: int = DEFAULT_MAX_CHARS, directory: Path | None = None,
                  save: bool = True) -> dict:
    """
    Ingesta un corpus: limpieza -> chunking por párrafos -> metadata -> embeddings en batch -> persistencia.

    embed_fn: función de embeddings; si es None se usa el proveedor de LLM_PROVIDER.
    directory: carpeta donde guardar el JSON; por defecto /embeddings.
    Devuelve {"path": Path | None, "chunks": list[Chunk], "vectors": list[list[float]]}.
    """
    if not documents:
        raise ValueError("The corpus is empty")

    seen_ids: set[str] = set()
    for doc in documents:
        if doc.document_id in seen_ids:
            raise ValueError(f"Duplicate document_id: '{doc.document_id}' (name '{doc.name}')")
        seen_ids.add(doc.document_id)

    ingestion_date = datetime.now(timezone.utc).isoformat()

    chunks: list[Chunk] = []
    for doc in documents:
        clean = clean_text(doc.text)
        if not clean:
            raise ValueError(f"Document '{doc.name}' is empty after cleaning")
        chunks.extend(build_chunks(doc, clean, ingestion_date, max_chars))

    if embed_fn is None:
        embed_fn = get_embed_fn()
        provider = get_provider_name()
    else:
        provider = "custom"

    vectors = embed_fn([c.text for c in chunks])

    path = None
    if save:
        path = save_embeddings(chunks, vectors, provider, directory)

    return {"path": path, "chunks": chunks, "vectors": vectors}
