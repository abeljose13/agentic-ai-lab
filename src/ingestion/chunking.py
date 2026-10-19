import hashlib
import re
from dataclasses import asdict, dataclass

from src.ingestion.corpus import Document

DEFAULT_MAX_CHARS = 1200  # margen por debajo del ~1500 exigido


@dataclass(frozen=True)
class Chunk:
    """Fragmento de un documento con su metadata de trazabilidad."""
    chunk_id: str
    document_id: str
    position: int          # orden del chunk dentro del documento (0-based)
    source: str            # URL o nombre original del documento
    ingestion_date: str    # timestamp ISO 8601 (UTC)
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long_block(block: str, max_chars: int) -> list[str]:
    """Parte un párrafo que supera max_chars: primero por oraciones, y como último recurso por caracteres."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_by_paragraphs(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """
    Estrategia por párrafos: agrupa párrafos completos (separados por línea en blanco)
    mientras el chunk no supere max_chars. Un párrafo que no entra solo se parte
    por oraciones. Ningún chunk excede max_chars.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    blocks: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            blocks.append(paragraph)
        else:
            blocks.extend(_split_long_block(paragraph, max_chars))

    chunks: list[str] = []
    current = ""
    for block in blocks:
        if not current:
            current = block
        elif len(current) + 2 + len(block) <= max_chars:
            current += "\n\n" + block
        else:
            chunks.append(current)
            current = block
    if current:
        chunks.append(current)
    return chunks


def build_chunks(document: Document, clean: str, ingestion_date: str,
                 max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """
    Construye los chunks de un documento ya limpio, con metadata completa.
    chunk_id = <document_id>-<posición 3 dígitos>-<hash del contenido>: determinístico
    y único dentro del corpus, y el índice del archivo de embeddings permite recuperar
    documento y posición a partir de él.
    """
    chunks = []
    for position, text in enumerate(chunk_by_paragraphs(clean, max_chars)):
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        chunks.append(Chunk(
            chunk_id=f"{document.document_id}-{position:03d}-{content_hash}",
            document_id=document.document_id,
            position=position,
            source=document.source,
            ingestion_date=ingestion_date,
            text=text,
        ))
    return chunks
