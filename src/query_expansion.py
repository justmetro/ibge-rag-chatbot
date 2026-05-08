def expand_query(question: str) -> str:
    """
    Expande perguntas do usuário com termos semanticamente úteis
    para melhorar a recuperação no banco vetorial.

    A pergunta original é preservada. Apenas adicionamos termos de busca
    relacionados ao tema detectado.
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