import os
from dotenv import load_dotenv

# Proveedores soportados por el laboratorio
VALID_PROVIDERS = ("openai", "anthropic", "google", "ollama", "groq")

_env_loaded = False


def load_env() -> None:
    """Carga el archivo .env una sola vez (idempotente)."""
    global _env_loaded
    if not _env_loaded:
        load_dotenv()
        _env_loaded = True


def get_provider_name() -> str:
    """Devuelve el proveedor configurado en LLM_PROVIDER (valor por defecto: 'google')."""
    load_env()
    provider = os.environ.get("LLM_PROVIDER", "google").lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' is not supported. Valid options: {list(VALID_PROVIDERS)}"
        )
    return provider
