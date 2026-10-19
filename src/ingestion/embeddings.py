import os
import time
from typing import Callable

import requests

from src.llm_client.config import get_provider_name
from src.llm_client.llm_client import TRANSIENT_ERRORS

EmbedFn = Callable[[list[str]], list[list[float]]]

BATCH_SIZE = 64
MAX_RETRIES = 3
_TRANSIENT_ERRORS = TRANSIENT_ERRORS + (requests.ConnectionError, requests.Timeout)


# =============================================================================
# PROVEEDORES: cada función recibe una lista de textos y devuelve un vector por texto
# =============================================================================

def embed_openai(texts: list[str]) -> list[list[float]]:
    """Embeddings con la API de OpenAI."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing environment variable 'OPENAI_API_KEY' for OpenAI embeddings.")
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-large"),
        input=texts,
    )
    return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]


def embed_google(texts: list[str]) -> list[list[float]]:
    """Embeddings con Google GenAI (`google-genai`)."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' for Google embeddings.")
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=os.environ.get("GOOGLE_EMBEDDING_MODEL_ID", "gemini-embedding-001"),
        contents=texts,
    )
    return [e.values for e in response.embeddings]


def embed_ollama(texts: list[str]) -> list[list[float]]:
    """Embeddings con un servidor Ollama (LAN), mediante su API REST /api/embed."""
    mac_server_ip = os.getenv("MAC_SERVER_IP", "192.168.1.14")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    payload = {
        "model": os.environ.get("OLLAMA_EMBEDDING_MODEL_ID", "nomic-embed-text"),
        "input": texts,
    }
    response = requests.post(f"http://{mac_server_ip}:{ollama_port}/api/embed", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["embeddings"]


EMBED_PROVIDERS: dict[str, Callable[[list[str]], list[list[float]]]] = {
    "openai": embed_openai,
    "google": embed_google,
    "ollama": embed_ollama,
}


# =============================================================================
# BATCH + REINTENTOS (común a todos los proveedores)
# =============================================================================

def _call_with_retries(provider_fn, batch: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            return provider_fn(batch)
        except _TRANSIENT_ERRORS as err:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Embeddings failed after {MAX_RETRIES} attempts: {err}") from err
            wait_seconds = 2 ** attempt
            print(f"[Embeddings] Transient error, retrying in {wait_seconds}s... Detail: {err}")
            time.sleep(wait_seconds)


def embed_texts(texts: list[str], provider_fn) -> list[list[float]]:
    """Embebe todos los textos en lotes de BATCH_SIZE, con reintentos ante errores transitorios."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_call_with_retries(provider_fn, texts[i:i + BATCH_SIZE]))
    if len(vectors) != len(texts):
        raise RuntimeError(f"Provider returned {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def get_embed_fn() -> EmbedFn:
    """
    Devuelve la función de embeddings del proveedor configurado en LLM_PROVIDER.
    Anthropic y Groq no ofrecen API de embeddings, por lo que no son válidos aquí.
    """
    provider = get_provider_name()
    if provider not in EMBED_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' does not offer an embeddings API. "
            f"Set LLM_PROVIDER to one of: {list(EMBED_PROVIDERS)}"
        )
    provider_fn = EMBED_PROVIDERS[provider]
    return lambda texts: embed_texts(texts, provider_fn)
