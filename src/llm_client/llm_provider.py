from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Interfaz común para cualquier proveedor de modelos de lenguaje."""

    @abstractmethod
    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Devuelve el texto generado por el modelo."""
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, schema_model, system: str = None):
        """Implementación por defecto: sirve para CUALQUIER proveedor, aunque no tenga
        soporte nativo de structured outputs. Un proveedor que sí lo tenga (ejemplo: Bedrock)
        puede sobreescribir este método con una versión más eficiente."""
        return generate_with_retries(self.generate, prompt, schema_model, system=system)