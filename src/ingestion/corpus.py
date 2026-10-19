import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CORPUS = PROJECT_ROOT / "docs" / "docs_examples_1.txt"

_RE_DOCUMENT = re.compile(r"^###DOCUMENT:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    """Documento de entrada del pipeline. `url` se usa como source si existe; si no, el nombre."""
    name: str
    text: str
    url: str | None = None

    @property
    def document_id(self) -> str:
        """Identificador estable derivado del nombre (slug ASCII)."""
        return slugify(self.name)

    @property
    def source(self) -> str:
        """Origen original del documento: URL si existe, o el nombre del archivo/texto."""
        return self.url or self.name


def slugify(text: str) -> str:
    """Convierte un nombre en un identificador ASCII en minúsculas separado por guiones."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if not slug:
        raise ValueError(f"Could not generate a document_id from '{text}'")
    return slug


def load_corpus(path: str | Path = EXAMPLE_CORPUS) -> list[Document]:
    """
    Lee un archivo de corpus con bloques delimitados por '###DOCUMENT: <nombre>'.
    Devuelve la lista de documentos en el orden del archivo.
    """
    content = Path(path).read_text(encoding="utf-8")
    matches = list(_RE_DOCUMENT.finditer(content))
    if not matches:
        raise ValueError(f"No '###DOCUMENT:' blocks found in {path}")

    documents = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[match.end():end].strip("\n")
        documents.append(Document(name=match.group(1), text=text))
    return documents
