import os
import time
import concurrent.futures
from langchain_openai import ChatOpenAI
from typing import List, Optional, Any
from dotenv import load_dotenv


class _NeverRaisedError(Exception):
    """Sentinela usada cuando un SDK no está instalado: nunca coincide en un except."""


# Intentar importar las excepciones de Rate Limit específicas de cada SDK
try:
    import openai
    OPENAI_RATE_LIMIT_ERR = openai.RateLimitError
except ImportError:
    openai = None
    OPENAI_RATE_LIMIT_ERR = _NeverRaisedError

try:
    import anthropic
    ANTHROPIC_RATE_LIMIT_ERR = anthropic.RateLimitError
except ImportError:
    anthropic = None
    ANTHROPIC_RATE_LIMIT_ERR = _NeverRaisedError

try:
    import groq
    GROQ_RATE_LIMIT_ERR = groq.RateLimitError
except ImportError:
    groq = None
    GROQ_RATE_LIMIT_ERR = _NeverRaisedError

# Errores que se consideran transitorios y por lo tanto reintentables
_TRANSIENT_ERRORS = [ConnectionError, TimeoutError]
if openai is not None:
    _TRANSIENT_ERRORS += [openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError]
if anthropic is not None:
    _TRANSIENT_ERRORS += [anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError]
if groq is not None:
    _TRANSIENT_ERRORS += [groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError]
TRANSIENT_ERRORS = tuple(_TRANSIENT_ERRORS)


class LLMClient:
    def __init__(self):
        load_dotenv()
        self.provider = os.environ.get("LLM_PROVIDER", "google").lower()

    def _validate_parameters(self, temperature: float, max_tokens: int) -> None:
        """Valida que los parámetros de generación estén en rangos válidos antes de llamar a los SDKs."""
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(f"Parámetro 'temperature' inválido ({temperature}). Debe estar entre 0.0 y 2.0.")
        if max_tokens <= 0:
            raise ValueError(f"Parámetro 'max_tokens' inválido ({max_tokens}). Debe ser un entero positivo mayor a 0.")

    def _execute_with_backoff(self, call_fn, *args, max_reintentos: int = 3, **kwargs) -> str:
        """
        Ejecuta la función de llamada al LLM aplicando el patrón Backoff Exponencial
        ante errores de Rate Limit o errores de red.

        Los errores no transitorios (validación, errores de programación, etc.)
        se relanzan inmediatamente sin reintentar.
        """
        for intento in range(max_reintentos):
            try:
                return call_fn(*args, **kwargs)
            except TRANSIENT_ERRORS as err:
                # Si es el último intento, lanzar un error claro al usuario
                if intento == max_reintentos - 1:
                    raise RuntimeError(f"Error persistente tras {max_reintentos} intentos con {self.provider}: {str(err)}") from err

                espera = 2 ** intento  # Backoff exponencial: 1s, 2s, 4s...
                print(f"[Rate Limit / Error en {self.provider}] Reintentando (intento {intento + 1}/{max_reintentos}) en {espera}s... Detalle: {err}")
                time.sleep(espera)
            except Exception:
                # Errores no reintentables (ValueError, TypeError, etc.): propagar tal cual
                raise

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Método principal para generar respuestas de texto desde el proveedor configurado.
        """
        # Validar parámetros de entrada
        self._validate_parameters(temperature, max_tokens)

        # Ruteo hacia la función correspondiente según self.provider
        providers_map = {
            "openai": self.call_openai,
            "anthropic": self.call_anthropic,
            "google": self.call_google,
            "ollama": self.call_ollama,
            "groq": self.call_groq,
        }

        call_fn = providers_map.get(self.provider)
        if not call_fn:
            raise ValueError(
                f"Proveedor '{self.provider}' no soportado. "
                f"Opciones válidas: {list(providers_map.keys())}"
            )

        # Ejecutar la llamada envuelta en el patrón de backoff exponencial
        return self._execute_with_backoff(
            call_fn,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def generate_batch(self, prompts: List[str], max_workers: int = 5) -> List[str]:
        """
        Resuelve una lista de prompts de forma concurrente utilizando ThreadPoolExecutor.
        """
        if not prompts:
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Mapear cada prompt a una ejecución individual de self.generate
            resultados = list(executor.map(self.generate, prompts))
        
        return resultados

    # =========================================================================
    # IMPLEMENTACIÓN DE PROVEEDORES
    # =========================================================================

    def call_openai(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        """Llamada usando el SDK oficial de OpenAI."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno 'OPENAI_API_KEY' para el proveedor OpenAI.")

        import openai
        client = openai.OpenAI(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        print("Ejecutando llamada a OpenAI - GPT")
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def call_anthropic(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        """Llamada usando el SDK oficial de Anthropic (Claude)."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno 'ANTHROPIC_API_KEY' para el proveedor Anthropic.")

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Claude limita la temperatura en el rango [0.0, 1.0]
        claude_temp = max(0.0, min(1.0, float(temperature)))
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

        # Preparar llamada explícita
        extra_args = {}
        if system:
            extra_args["system"] = system

        print("Ejecutando llamada a Anthropic - Claude")
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            extra_body={"temperature": claude_temp},
            messages=[{"role": "user", "content": prompt}],
            **extra_args
        )
        return response.content[0].text.strip()

    def call_google(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        """Llamada usando el nuevo SDK de Google GenAI (`google-genai`)."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno 'GEMINI_API_KEY' o 'GOOGLE_API_KEY' para el proveedor Google.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system if system else None
        )

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        print("Ejecutando llamada a Google - Gemini")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return response.text.strip()

    def call_groq(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        """Llamada usando el SDK oficial de Groq."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno 'GROQ_API_KEY' para el proveedor Groq.")

        import groq
        client = groq.Groq(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model_name = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        print("Ejecutando llamada a Groq")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def call_ollama_rest(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        """Llamada a servidor Ollama local mediante peticiones HTTP/REST."""
        import requests

        MAC_SERVER_IP = os.getenv("MAC_SERVER_IP", "192.168.1.14")
        OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
        OLLAMA_BASE_URL = f"http://{MAC_SERVER_IP}:{OLLAMA_PORT}"

        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if system:
            payload["system"] = system

        print("Ejecutando llamada a Ollama")
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def call_ollama(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        """Llamada a servidor Ollama local mediante langchain openai."""
        MAC_SERVER_IP = os.getenv("MAC_SERVER_IP", "192.168.1.14")
        OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
        OLLAMA_BASE_URL = f"http://{MAC_SERVER_IP}:{OLLAMA_PORT}/v1"

        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

        llm = self.get_client_llm(temperature, max_tokens, OLLAMA_BASE_URL, model)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        print("Ejecutando llamada a Ollama")
        response = llm.invoke(messages)

        return response.content.strip()

    def get_client_llm(self, temperature: float, max_tokens: int, ollama_url: str, model: str):
        """ 
        Crea un cliente para consumir un modelo en el servidor Ollama local. 
        """
        return ChatOpenAI(
            base_url=ollama_url,
            api_key="ollama",
            model=model,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )
        