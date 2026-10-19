import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.chunking import Chunk
from src.ingestion.corpus import PROJECT_ROOT

EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"


def save_embeddings(chunks: list[Chunk], vectors: list[list[float]], provider: str,
                    directory: Path | None = None) -> Path:
    """
    Guarda chunks, vectores y el índice chunk_id -> (document_id, position, source)
    en un JSON <run_id>_<fecha>.json dentro de /embeddings.
    Migración futura: este JSON se reemplazará por un vector store real.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"There are {len(chunks)} chunks but {len(vectors)} vectors")

    directory = directory or EMBEDDINGS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    created = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:8]
    path = directory / f"{run_id}_{created:%Y%m%d_%H%M%S}.json"

    data = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "provider": provider,
        "total_chunks": len(chunks),
        "chunks": [{**c.to_dict(), "vector": v} for c, v in zip(chunks, vectors)],
        "index": {
            c.chunk_id: {"document_id": c.document_id, "position": c.position, "source": c.source}
            for c in chunks
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def find_chunk(path: Path, chunk_id: str) -> dict:
    """Dado un chunk_id, devuelve {document_id, position, source} desde el índice del archivo."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if chunk_id not in data["index"]:
        raise KeyError(f"chunk_id '{chunk_id}' not found in {path}")
    return data["index"][chunk_id]
