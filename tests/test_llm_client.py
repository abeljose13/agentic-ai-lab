from src.llm_client.llm_client import LLMClient

client = LLMClient()

# Prueba básica (prompt de entrada en español, a propósito)
response = client.generate("¿Cuál es la capital de Francia?")
print("Response:", response)

# Prueba batch
responses = client.generate_batch([
    "¿Cuál es la capital de España?",
    "¿Cuál es la capital de Italia?"
])
print("Batch:", responses)

# Prueba de error claro
try:
    client.generate("Hola", max_tokens=-10)
except ValueError as e:
    print("Validation error caught correctly:", e)
