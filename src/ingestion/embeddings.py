import os
import time
from typing import Callable

import requests

from src.llm_client.config import get_provider_name, load_env
from src.llm_client.llm_client import OPENROUTER_BASE_URL, TRANSIENT_ERRORS

# Función de embeddings: recibe textos y, opcionalmente, input_type ("search_document" o "search_query").
EmbedFn = Callable[..., list[list[float]]]

BATCH_SIZE = 64
MAX_RETRIES = 3


class ProviderTransientError(Exception):
    """Error HTTP transitorio de un proveedor REST (429 rate limit, 529 overloaded): se reintenta."""


_TRANSIENT_ERRORS = TRANSIENT_ERRORS + (requests.ConnectionError, requests.Timeout, ProviderTransientError)
_RETRYABLE_HTTP_STATUS = (429, 529)

# Valores válidos de input_type: documento a indexar vs. consulta del usuario.
VALID_INPUT_TYPES = ("search_document", "search_query")

# Traducción de input_type a task_type de Google GenAI (los otros proveedores lo ignoran).
GOOGLE_TASK_TYPES = {"search_document": "RETRIEVAL_DOCUMENT", "search_query": "RETRIEVAL_QUERY"}

# Modelo por defecto y variable de entorno que lo sobreescribe, por proveedor.
EMBED_MODEL_SETTINGS = {
    "openai": ("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-large"),
    "google": ("GOOGLE_EMBEDDING_MODEL_ID", "gemini-embedding-001"),
    "ollama": ("OLLAMA_EMBEDDING_MODEL_ID", "bge-m3"),
    "openrouter": ("OPENROUTER_EMBEDDING_MODEL_ID", "nvidia/llama-nemotron-embed-vl-1b-v2:free"),
}


# =============================================================================
# PROVEEDORES: cada función recibe textos e input_type y devuelve un vector por texto
# =============================================================================

def embed_openai(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embeddings con la API de OpenAI. Los modelos text-embedding no distinguen input_type: se ignora."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing environment variable 'OPENAI_API_KEY' for OpenAI embeddings.")
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=os.environ.get(*EMBED_MODEL_SETTINGS["openai"]),
        input=texts,
    )
    return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]


def embed_google(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embeddings con Google GenAI (`google-genai`). input_type se traduce a task_type."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' for Google embeddings.")
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=os.environ.get(*EMBED_MODEL_SETTINGS["google"]),
        contents=texts,
        config=types.EmbedContentConfig(task_type=GOOGLE_TASK_TYPES[input_type]),
    )
    return [e.values for e in response.embeddings]


def embed_ollama(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embeddings con un servidor Ollama, mediante su API REST /api/embed.
    nomic-embed-text y mxbai-embed-large no distinguen input_type: se ignora."""
    mac_server_ip = os.getenv("MAC_SERVER_IP", "192.168.1.14")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    payload = {
        "model": os.environ.get(*EMBED_MODEL_SETTINGS["ollama"]),
        "input": texts,
    }
    response = requests.post(f"http://{mac_server_ip}:{ollama_port}/api/embed", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_openrouter(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embeddings con OpenRouter (API OpenAI-compatible). input_type se envía tal cual: OpenRouter lo soporta."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing environment variable 'OPENROUTER_API_KEY' for OpenRouter embeddings.")
    payload = {
        "model": os.environ.get(*EMBED_MODEL_SETTINGS["openrouter"]),
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
    }
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if response.status_code in _RETRYABLE_HTTP_STATUS:
        raise ProviderTransientError(f"OpenRouter returned HTTP {response.status_code}: {response.text[:200]}")
    response.raise_for_status()
    return [d["embedding"] for d in sorted(response.json()["data"], key=lambda d: d["index"])]


EMBED_PROVIDERS: dict[str, Callable[..., list[list[float]]]] = {
    "openai": embed_openai,
    "google": embed_google,
    "ollama": embed_ollama,
    "openrouter": embed_openrouter,
}


# =============================================================================
# BATCH + REINTENTOS (común a todos los proveedores)
# =============================================================================

def _call_with_retries(provider_fn, batch: list[str], input_type: str) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            return provider_fn(batch, input_type=input_type)
        except _TRANSIENT_ERRORS as err:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Embeddings failed after {MAX_RETRIES} attempts: {err}") from err
            wait_seconds = 2 ** attempt
            print(f"[Embeddings] Transient error, retrying in {wait_seconds}s... Detail: {err}")
            time.sleep(wait_seconds)


def embed_texts(texts: list[str], provider_fn, input_type: str = "search_document") -> list[list[float]]:
    """Embebe todos los textos en lotes de BATCH_SIZE, con reintentos ante errores transitorios."""
    if input_type not in VALID_INPUT_TYPES:
        raise ValueError(f"Invalid input_type '{input_type}'. Valid options: {list(VALID_INPUT_TYPES)}")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_call_with_retries(provider_fn, texts[i:i + BATCH_SIZE], input_type))
    if len(vectors) != len(texts):
        raise RuntimeError(f"Provider returned {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def get_embed_fn() -> EmbedFn:
    """
    Devuelve la función de embeddings del proveedor configurado en LLM_PROVIDER.
    Anthropic y Groq no ofrecen API de embeddings, por lo que no son válidos aquí.
    La función devuelta acepta (texts, input_type="search_document").
    """
    provider = get_provider_name()
    if provider not in EMBED_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' does not offer an embeddings API. "
            f"Set LLM_PROVIDER to one of: {list(EMBED_PROVIDERS)}"
        )
    provider_fn = EMBED_PROVIDERS[provider]
    return lambda texts, input_type="search_document": embed_texts(texts, provider_fn, input_type)


def get_embed_model(provider: str | None = None) -> str:
    """Devuelve el id del modelo de embeddings configurado para `provider` (por defecto, LLM_PROVIDER)."""
    load_env()
    provider = provider or get_provider_name()
    if provider not in EMBED_MODEL_SETTINGS:
        raise ValueError(f"Provider '{provider}' does not offer an embeddings API.")
    env_var, default = EMBED_MODEL_SETTINGS[provider]
    return os.environ.get(env_var, default)
