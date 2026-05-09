"""
Query expansion utilities for the RAG retrieval step.

This module expands user questions with additional domain-specific terms
before sending them to the vector database. The goal is to improve recall
for common themes in the IBGE tables, such as income, sex, race/color,
territorial cuts and coefficient-of-variation tables.

The original user question is preserved. Extra terms are appended only to
help the retriever find better chunks.
"""


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
        'Compare homens e mulheres sexo homem mulher rendimento trabalho principal'
    """
    question_lower = question.lower()
    extra_terms = []

    asks_for_values = (
        "valor" in question_lower
        or "valores" in question_lower
        or "quanto" in question_lower
        or "r$" in question_lower
        or "reais" in question_lower
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

    if (
        "homem" in question_lower
        or "homens" in question_lower
        or "mulher" in question_lower
        or "mulheres" in question_lower
    ):
        extra_terms.extend([
            "sexo",
            "homem",
            "mulher",
            "rendimento",
            "trabalho principal",
        ])

    if "cor" in question_lower or "raça" in question_lower or "raca" in question_lower:
        extra_terms.extend([
            "cor ou raça",
            "branca",
            "preta",
            "parda",
            "rendimento",
        ])

    if "rondônia" in question_lower or "rondonia" in question_lower:
        extra_terms.extend([
            "Brasil",
            "Norte",
            "Rondônia",
            "Grandes Regiões",
            "Unidades da Federação",
        ])

    if "rendimento" in question_lower:
        extra_terms.extend([
            "rendimento médio real habitual",
            "rendimento-hora",
            "R$/mês",
            "R$/hora",
        ])

    if (
        "coeficiente" in question_lower
        or "coeficientes" in question_lower
        or "variação" in question_lower
        or "variacao" in question_lower
        or "cv" in question_lower
    ):
        extra_terms.extend([
            "coeficientes de variação",
            "coeficiente de variação dos indicadores",
        ])

    if not extra_terms:
        return question

    return question + " " + " ".join(extra_terms)