import os

class LLMClient:
    def __init__(self):
        load_dotenv()
        self.provider = os.environ.get("LLM_PROVIDER", "openai")

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        # TODO: según self.provider, llamar a la función  call_anthropic /
        # call_openai / call_google / call_ollama correspondiente, envuelta en manejo de reintentos/errores.
        # Cantidad de reintentos 3. Manejo de errores eficiente
        raise NotImplementedError("Completar generate()")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        # TODO (opcional): resolver la lista de prompts de forma concurrente
        raise NotImplementedError("Completar generate_batch() (opcional)")
    