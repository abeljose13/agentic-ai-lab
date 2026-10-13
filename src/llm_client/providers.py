import os
from typing import TypeVar
from pydantic import BaseModel

from src.llm_client.llm_provider import LLMProvider
from src.llm_client.llm_client import LLMClient

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = LLMClient()
        self.client.provider = "anthropic"

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        return self.client.generate(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = LLMClient()
        self.client.provider = "openai"

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        return self.client.generate(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)


class GoogleProvider(LLMProvider):
    def __init__(self):
        self.client = LLMClient()
        self.client.provider = "google"

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        return self.client.generate(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)


class GroqProvider(LLMProvider):
    def __init__(self):
        self.client = LLMClient()
        self.client.provider = "groq"

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        return self.client.generate(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)


class OllamaProvider(LLMProvider):
    def __init__(self):
        self.client = LLMClient()
        self.client.provider = "ollama"

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        return self.client.generate(prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)


def get_provider() -> LLMProvider:
    """Factory: devuelve la instancia del proveedor configurado en LLM_PROVIDER."""
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "groq": GroqProvider,
        "ollama": OllamaProvider,
    }
    name = os.environ.get("LLM_PROVIDER", "google").lower()
    if name not in providers:
        raise ValueError(f"Unknown provider: '{name}'. Opciones: {list(providers.keys())}")
    return providers[name]()
