def expand_query(question: str) -> str:
    """
    Expande perguntas do usuário com termos semanticamente úteis
    para melhorar a recuperação no banco vetorial.

    Esta etapa é simples e transparente: não altera a pergunta original,
    apenas adiciona termos de busca quando detecta temas específicos.
    """
    question_lower = question.lower()
    extra_terms = []

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

    if not extra_terms:
        return question

    expanded = question + " " + " ".join(extra_terms)

    return expanded