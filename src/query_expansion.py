"""
Query expansion utilities for the RAG retrieval step.

This module expands user questions with additional domain-specific terms
before sending them to the vector database. The goal is to improve recall
for common themes in the IBGE tables, such as income, sex, race/color,
territorial cuts and coefficient-of-variation tables.

The original user question is preserved. Extra terms are appended only to
help the retriever find better chunks.

Keywords are matched as whole words, ignoring accents: "cor" must not match
"recortes", otherwise territorial questions get race/color terms appended.
"""

from src.document_filters import user_wants_coefficient
from src.text_utils import contains_any_term


def expand_query(question: str) -> str:
    """
    Expand a user question with semantically useful search terms.

    The expansion is intentionally rule-based and transparent. It helps the
    retriever find relevant table chunks when the user's wording is shorter
    or different from the wording used in the original IBGE tables.

    Args:
        question: Original user question.

    Returns:
        The original question plus extra terms when a known topic is detected.
        If no topic is detected, returns the original question unchanged.

    Examples:
        >>> expand_query("Compare homens e mulheres")
        'Compare homens e mulheres sexo homem mulher rendimento trabalho principal indicadores segundo sexo'
    """
    extra_terms = []

    asks_for_values = contains_any_term(
        question,
        ["valor", "valores", "quanto", "r$", "reais"],
    )

    if asks_for_values:
        extra_terms.extend([
            "Indicadores de rendimento do trabalho",
            "rendimento médio real habitual do trabalho principal",
            "valores monetários",
            "R$/mês",
            "Brasil",
            "Norte",
            "Rondônia",
        ])

    if contains_any_term(
        question,
        ["sexo", "gênero", "homem", "homens", "mulher", "mulheres"],
    ):
        extra_terms.extend([
            "sexo",
            "homem",
            "mulher",
            "rendimento",
            "trabalho principal",
            "indicadores segundo sexo",
        ])

    if contains_any_term(question, ["cor", "raça"]):
        extra_terms.extend([
            "cor ou raça",
            "branca",
            "preta",
            "parda",
            "rendimento",
        ])

    if contains_any_term(
        question,
        ["territorial", "territoriais", "região", "regiões", "estado", "estados", "uf", "ufs"],
    ):
        extra_terms.extend([
            "Brasil",
            "Grandes Regiões",
            "Unidades da Federação",
        ])

    if contains_any_term(question, ["rondônia"]):
        extra_terms.extend([
            "Brasil",
            "Norte",
            "Rondônia",
            "Grandes Regiões",
            "Unidades da Federação",
        ])

    if contains_any_term(question, ["rendimento", "rendimentos"]):
        extra_terms.extend([
            "rendimento médio real habitual",
            "rendimento-hora",
            "R$/mês",
            "R$/hora",
        ])

    if user_wants_coefficient(question):
        extra_terms.extend([
            "coeficientes de variação",
            "coeficiente de variação dos indicadores",
        ])

    if not extra_terms:
        return question

    return question + " " + " ".join(extra_terms)