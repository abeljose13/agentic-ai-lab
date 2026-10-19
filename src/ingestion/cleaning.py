import re
import unicodedata

# Patrones de líneas que son encabezados/pies de página y no aportan contenido
# (coinciden contra el contenido del corpus, que está en español)
_BOILERPLATE_LINES = [
    re.compile(r"^\s*p[áa]gina\s+\d+\s+de\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*p[áa]g\.?\s*\d+\s*$", re.IGNORECASE),
]

# Mojibake típico (UTF-8 leído como Latin-1/CP1252) y su corrección
_MOJIBAKE = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\u009d": '"',
    "â€“": "-",
    "â€”": "-",
    "Ã¡": "á",
    "Ã©": "é",
    "Ã­": "í",
    "Ã³": "ó",
    "Ã±": "ñ",
    "Ã¼": "ü",
    "Ãº": "ú",
}

# Caracteres invisibles: zero-width, BOM, soft hyphen
_INVISIBLES = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")
# Caracteres de control (salvo \n y \t, que se tratan explícitamente)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean_text(text: str) -> str:
    """
    Limpia el texto crudo de un documento antes del chunking:
    - normaliza unicode (NFKC: NBSP, ligaduras, ancho completo)
    - corrige mojibake comunes y elimina caracteres invisibles/de control
    - quita encabezados y pies de página tipo 'Página X de Y'
    - une saltos de línea simples dentro de un párrafo (cortes de PDF)
    - colapsa espacios repetidos y saltos de línea excesivos
    """
    # Mojibake primero: NFKC transforma '™' (parte de 'â€™') en 'TM' y rompería la corrección
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLES.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _CONTROL.sub(" ", text)

    lines = [
        line for line in text.split("\n")
        if not any(pattern.match(line) for pattern in _BOILERPLATE_LINES)
    ]
    text = "\n".join(lines)

    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s*\n\s*", " ", block)   # salto simple dentro de párrafo -> espacio
        paragraph = re.sub(r"[ ]{2,}", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)
