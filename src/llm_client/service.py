from typing import Literal
from pydantic import BaseModel, Field
from src.llm_client.providers import get_provider


class IntentClassifierModel(BaseModel):
    """Modelo Pydantic para clasificar la intención de preguntas sobre documentación técnica."""

    question_type: Literal[
        "conceptual",
        "installation_configuration",
        "error_resolution",
        "out_of_domain"
    ] = Field(
        description="Main category of the query about the technical documentation."
    )
    requires_search_documentation: bool = Field(
        description="True if the question requires querying the RAG/technical documentation base."
    )
    summary: str = Field(
        description="Brief and clear summary of the question's goal in 1 or 2 sentences."
    )


def classify(text: str) -> IntentClassifierModel:
    """Transforma texto libre en un objeto IntentClassifierModel validado por Pydantic."""
    provider = get_provider()
    system_instruction = (
        "You are an expert intent classifier for a RAG system over technical documentation. "
        "Analyze the user's question and extract the required information concisely."
    )
    return provider.generate_structured(
        prompt=text,
        schema_model=IntentClassifierModel,
        system=system_instruction
    )
