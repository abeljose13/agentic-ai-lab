import os
from dotenv import load_dotenv

# Proveedores soportados por el laboratorio
PROVIDERS_VALIDOS = ("openai", "anthropic", "google", "ollama", "groq")

_env_cargado = False


def load_env() -> None:
    """Carga el archivo .env una sola vez (idempotente)."""
    global _env_cargado
    if not _env_cargado:
        load_dotenv()
        _env_cargado = True


def get_provider_name() -> str:
    """Devuelve el proveedor configurado en LLM_PROVIDER (valor por defecto: 'google')."""
    load_env()
    proveedor = os.environ.get("LLM_PROVIDER", "google").lower()
    if proveedor not in PROVIDERS_VALIDOS:
        raise ValueError(
            f"Proveedor '{proveedor}' no soportado. Opciones válidas: {list(PROVIDERS_VALIDOS)}"
        )
    return proveedor
