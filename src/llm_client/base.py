from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

class BaseLLMClient(ABC):
    """Interfaz base agnóstica para clientes LLM."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Genera una respuesta de texto plano a partir de un prompt."""
        pass

    @abstractmethod
    def generate_structured(
        self, prompt: str, response_schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        """Genera una respuesta estructurada garantizada mediante un esquema Pydantic."""
        pass