import json
import time
from abc import ABC, abstractmethod
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def generate_with_retries(
    generate_fn,
    prompt: str,
    schema_model: Type[T],
    system: str = None,
    max_retries: int = 3
) -> T:
    """
    Función agnóstica de fallback/reintento:
    Solicita JSON que cumpla el esquema de Pydantic y reintenta si el parseo falla.
    """
    json_schema = json.dumps(schema_model.model_json_schema(), indent=2)
    structured_system_prompt = (
        f"{system + '\n\n' if system else ''}"
        f"You MUST respond EXCLUSIVELY with a valid JSON that strictly complies "
        f"with the following JSON Schema:\n{json_schema}\n"
        f"Do not include extra markdown formatting or explanations outside the JSON."
    )

    last_error = None
    last_response = ""

    for attempt in range(max_retries):
        current_prompt = prompt
        if attempt > 0:
            current_prompt = (
                f"{prompt}\n\n[ATTENTION]: Your previous attempt failed validation.\n"
                f"Previous response: {last_response}\n"
                f"Validation error: {last_error}\n"
                f"Please correct the response so that it strictly complies with the schema."
            )

        raw_text = generate_fn(prompt=current_prompt, system=structured_system_prompt)
        last_response = raw_text

        # Limpiar posible bloque markdown ```json ... ```
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            return schema_model.model_validate_json(cleaned_text)
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = str(e)
            print(f"[Pydantic Validation Error - Attempt {attempt + 1}/{max_retries}]: {e}")
            time.sleep(1)

    raise ValueError(
        f"A valid object could not be obtained for {schema_model.__name__} "
        f"after {max_retries} attempts. Last error: {last_error}"
    )


class LLMProvider(ABC):
    """Interfaz común para cualquier proveedor de modelos de lenguaje."""

    @abstractmethod
    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Devuelve el texto generado por el modelo."""
        pass

    def generate_structured(self, prompt: str, schema_model: Type[T], system: str = None) -> T:
        """
        Implementación por defecto usando reintento + prompt.
        Aplica para proveedores sin API nativa de Structured Outputs.
        """
        return generate_with_retries(self.generate, prompt, schema_model, system=system)
