"""
Tests offline del pipeline de ingesta (sin llamadas a APIs de embeddings).
Ejecutar desde la raíz: uv run python tests/test_ingestion_offline.py
"""
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz del repo para imports src.*

from src.ingestion.chunking import DEFAULT_MAX_CHARS, chunk_by_paragraphs
from src.ingestion.cleaning import clean_text
from src.ingestion.corpus import EXAMPLE_CORPUS, Document, load_corpus
from src.ingestion.pipeline import run_ingestion
from src.ingestion.storage import find_chunk

CHUNK_LIMIT = 1500  # criterio de éxito: "menos de ~1500 caracteres"
METADATA_FIELDS = {"chunk_id", "document_id", "position", "source", "ingestion_date", "text"}


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Vector determinístico de dimensión fija, sin llamadas externas."""
    return [[float(len(t)), float(sum(map(ord, t)) % 997), float(t.count(" "))] for t in texts]


def test_corpus_meets_requirements():
    docs = load_corpus()
    assert len(docs) == 3, f"Expected 3 documents, found {len(docs)}"
    assert len({d.name for d in docs}) == 3, "Names must be distinct"
    for d in docs:
        assert len(d.text) <= 6000, f"'{d.name}' exceeds 6000 characters"
        paragraphs = [p for p in d.text.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 3, f"'{d.name}' has fewer than 3 paragraphs"
        noise = re.search(r"[ \t]{2,}|\n{3,}|Página \d+ de \d+|â€|\u00a0|\u200b", d.text)
        assert noise, f"'{d.name}' has no noise characters to exercise the cleaning step"


def test_clean_text_removes_noise():
    for doc in load_corpus():
        cleaned = clean_text(doc.text)
        assert not re.search(r"Página \d+ de \d+", cleaned, re.IGNORECASE), "Page header left behind"
        assert "\u200b" not in cleaned and "\u00a0" not in cleaned, "Invisible characters left behind"
        assert "â€" not in cleaned, "Mojibake left behind"
        assert "  " not in cleaned, "Repeated spaces left behind"
        assert "\n\n\n" not in cleaned, "Excessive line breaks left behind"
        assert cleaned == cleaned.strip(), "Leading or trailing whitespace remains"


def test_clean_text_specific_cases():
    assert clean_text("ﬁnanzas   con\u00a0espacio") == "finanzas con espacio"
    assert clean_text("Hola â€™mundoâ€™") == "Hola 'mundo'"
    assert clean_text("Línea uno\ncortada por PDF") == "Línea uno cortada por PDF"


def test_chunking_respects_limit_and_paragraphs():
    text = ("Párrafo corto uno.\n\nPárrafo corto dos.\n\n" + ("Oración larga. " * 200)).strip()
    chunks = chunk_by_paragraphs(text, max_chars=300)
    assert all(len(c) <= 300 for c in chunks), "A chunk exceeds max_chars"
    assert chunks[0].startswith("Párrafo corto uno.\n\nPárrafo corto dos."), "Short paragraphs were not grouped"


def test_chunks_metadata_and_limit():
    result = run_ingestion(load_corpus(), embed_fn=fake_embed, save=False)
    chunks = result["chunks"]
    assert chunks, "No chunks were generated"

    for c in chunks:
        assert len(c.text) < CHUNK_LIMIT, f"Chunk {c.chunk_id} has {len(c.text)} characters"
        assert c.text.strip(), "Empty chunk"

    for doc in load_corpus():
        own = [c for c in chunks if c.document_id == doc.document_id]
        assert own, f"'{doc.name}' generated no chunks"
        assert [c.position for c in own] == list(range(len(own))), "Positions are not sequential"
        assert all(c.source == doc.name for c in own), "Wrong source"
        for c in own:
            datetime.fromisoformat(c.ingestion_date)  # lanza si no es timestamp válido
            assert METADATA_FIELDS <= set(c.to_dict().keys()), "Missing metadata"

    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_id is not unique"


def test_chunk_id_is_deterministic():
    a = run_ingestion(load_corpus(), embed_fn=fake_embed, save=False)["chunks"]
    b = run_ingestion(load_corpus(), embed_fn=fake_embed, save=False)["chunks"]
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b], "chunk_id changes between runs"


def test_traceability_from_json_file():
    with tempfile.TemporaryDirectory() as tmp:
        result = run_ingestion(load_corpus(), embed_fn=fake_embed, directory=Path(tmp))
        path = result["path"]
        assert path is not None and path.exists(), "Embeddings file was not saved"
        assert path.name.endswith(".json") and path.parent == Path(tmp)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_chunks"] == len(result["chunks"]) == len(data["chunks"])
        for record in data["chunks"]:
            assert len(record["vector"]) == 3, "Unexpected vector dimension"
            assert record["text"], "Empty text saved"

        for c in result["chunks"]:
            ref = find_chunk(path, c.chunk_id)
            assert ref == {"document_id": c.document_id, "position": c.position, "source": c.source}, \
                f"Wrong traceability for {c.chunk_id}"


def test_empty_document_and_duplicates_fail_with_message():
    try:
        run_ingestion([Document(name="Empty", text="   \n\n  ")], embed_fn=fake_embed, save=False)
        raise AssertionError("Expected failure on empty document")
    except ValueError as err:
        assert "empty after cleaning" in str(err)
    try:
        run_ingestion([Document("Same", "a"), Document("same", "b")], embed_fn=fake_embed, save=False)
        raise AssertionError("Expected failure on duplicate document_id")
    except ValueError as err:
        assert "Duplicate" in str(err)


TESTS = [
    test_corpus_meets_requirements,
    test_clean_text_removes_noise,
    test_clean_text_specific_cases,
    test_chunking_respects_limit_and_paragraphs,
    test_chunks_metadata_and_limit,
    test_chunk_id_is_deterministic,
    test_traceability_from_json_file,
    test_empty_document_and_duplicates_fail_with_message,
]

if __name__ == "__main__":
    print(f"=== OFFLINE INGESTION TESTS (corpus: {EXAMPLE_CORPUS.name}) ===\n")
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as err:
            failures += 1
            print(f"FAIL  {test.__name__}: {err!r}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests OK")
    sys.exit(1 if failures else 0)
