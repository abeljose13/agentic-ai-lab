from typing import Literal
from pydantic import BaseModel, Field
from src.llm_client.providers import get_provider


class IntentClassifierModel(BaseModel):
    """Modelo Pydantic para clasificar la intención de preguntas sobre documentación técnica."""
    
    question_type: Literal[
        "conceptual", 
        "configuracion_instalacion", 
        "resolucion_errores", 
        "fuera_de_dominio"
    ] = Field(
        description="Categoría principal de la consulta sobre la documentación técnica."
    )
    requires_search_documentation: bool = Field(
        description="True si la pregunta requiere consultar la base RAG/documentación técnica."
    )
    summary: str = Field(
        description="Resumen breve y claro del objetivo de la pregunta en 1 o 2 oraciones."
    )


def classify(text: str) -> IntentClassifierModel:
    """Transforma texto libre en un objeto IntentClassifierModel validado por Pydantic."""
    provider = get_provider()
    system_instruction = (
        "Eres un clasificador experto de intenciones para un sistema RAG de documentación técnica. "
        "Analiza la pregunta del usuario y extrae la información requerida de forma concisa."
    )
    return provider.generate_structured(
        prompt=text,
        schema_model=IntentClassifierModel,
        system=system_instruction
    )
