"""
Almacenamiento vectorial con ChromaDB persistente.

Carga los chunks + embeddings generados por el pipeline de ingesta (guardados como
JSON en /embeddings) a una colección persistente, sin volver a embeber nada. La
separación ingesta/consulta queda explícita: la ingesta corre una vez y persiste;
la consulta solo abre el índice ya construido.

La colección guarda el modelo de embeddings que generó sus vectores. Si el run
cambia de modelo, la colección se reconstruye automáticamente.
"""
import json
from pathlib import Path

import chromadb

from src.ingestion.corpus import PROJECT_ROOT
from src.ingestion.embeddings import get_embed_model
from src.ingestion.storage import EMBEDDINGS_DIR

# Directorio local de la base vectorial persistente (ChromaDB).
CHROMA_DIR = PROJECT_ROOT / "chroma_data"
COLLECTION_NAME = "corpus_curso"


def latest_run_path(directory: Path = EMBEDDINGS_DIR) -> Path:
    """Devuelve el JSON de embeddings más reciente dentro de `directory`."""
    runs = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No embeddings runs found in {directory}")
    return runs[-1]


def load_embeddings_run(path: Path | str) -> tuple[list[dict], str | None]:
    """
    Lee un JSON de /embeddings y devuelve (chunks, modelo). Cada chunk trae su
    metadata y su vector ya generado. El modelo es None en runs anteriores a
    que el JSON registrara el campo "model".
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = data["chunks"]
    for c in chunks:
        if not c.get("vector"):
            raise ValueError(f"Chunk '{c['chunk_id']}' has no vector in {path}")
    return chunks, data.get("model")


def collection_embedding_model(collection) -> str | None:
    """Modelo de embeddings con el que se construyó la colección (None si no está registrado)."""
    return (collection.metadata or {}).get("embedding_model")


def _upsert_chunks(collection, chunks: list[dict]) -> None:
    """Escribe chunks en la colección. `upsert` por chunk_id: idempotente."""
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["vector"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "document_id": c["document_id"],
                "position": c["position"],
                "source": c["source"],
                "ingestion_date": c["ingestion_date"],
            }
            for c in chunks
        ],
    )


def build_or_open_collection(embeddings_path: Path | str | None = None,
                             collection_name: str = COLLECTION_NAME,
                             db_path: Path = CHROMA_DIR) -> "chromadb.api.Collection.Collection":
    """
    Abre (o crea) la colección persistente y asegura que contenga los chunks del run.

    - Si el modelo registrado en la colección no coincide con el del run, la
      colección se borra y se reconstruye (sus vectores son de otro espacio).
    - Si coincide, hace upsert igualmente: es idempotente por chunk_id, no re-embebe
      y cubre el caso de un corpus editado re-ingerido con el mismo modelo.
    """
    embeddings_path = Path(embeddings_path) if embeddings_path else latest_run_path()
    chunks, run_model = load_embeddings_run(embeddings_path)
    # Runs sin campo "model": se asume el modelo configurado actualmente.
    run_model = run_model or get_embed_model()

    client = chromadb.PersistentClient(path=str(db_path))
    metadata = {"hnsw:space": "cosine", "embedding_model": run_model}
    collection = client.get_or_create_collection(name=collection_name, metadata=metadata)

    if collection_embedding_model(collection) != run_model:
        print(f"[Index] Collection model '{collection_embedding_model(collection)}' != run model "
              f"'{run_model}': rebuilding collection '{collection_name}'")
        client.delete_collection(collection_name)
        collection = client.create_collection(name=collection_name, metadata=metadata)

    # Upsert incondicional: idempotente por chunk_id y no re-embebe nada.
    _upsert_chunks(collection, chunks)

    return collection
