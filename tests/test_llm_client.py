from src.llm_client.llm_client import LLMClient

client = LLMClient()

# Prueba básica
respuesta = client.generate("¿Cuál es la capital de Francia?")
print("Respuesta:", respuesta)

# Prueba batch
respuestas = client.generate_batch([
    "¿Cuál es la capital de España?",
    "¿Cuál es la capital de Italia?"
])
print("Batch:", respuestas)

# Prueba de error claro
try:
    client.generate("Hola", max_tokens=-10)
except ValueError as e:
    print("Error de validación capturado correctamente:", e)