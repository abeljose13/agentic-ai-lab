from src.llm_client.service import classify, IntentClassifierModel

# 3 entradas de prueba
test_cases = [
    # 1. Caso claro técnico / conceptual
    "¿Qué diferencia hay entre la arquitectura síncrona y asíncrona de la API y cómo funciona el pipeline RAG?",
    
    # 2. Caso de error / configuración
    "Tengo un error HTTP 500 al llamar al endpoint /v1/chat con la API key de Gemini, ¿cómo lo arreglo?",
    
    # 3. Caso ambiguo / mal escrito / fuera de tema
    "ola keria saber si me podes decir receta para pizza o como configuro la base de datos xq no anda nada..."
]

if __name__ == "__main__":
    print("=== TESTING CHECKPOINT CLASS 3 (STRUCTURED OUTPUT) ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"--- Case {i} ---")
        print(f"Input: {text}")
        
        try:
            result = classify(text)
            print("\nResult Validated by Pydantic:")
            print(f" Question type: {result.question_type}")
            print(f" Requires search: {result.requires_search_documentation}")
            print(f" Summary: {result.summary}")
            print(f" Python Object Type: {type(result)}")
            assert isinstance(result, IntentClassifierModel)
            print(" Validation SUCCESS: It is a strict instance of IntentClassifierModel.")
        except Exception as e:
            print(f" FAIL: An explicit exception was thrown: {e}")
            
        print("-" * 60)
