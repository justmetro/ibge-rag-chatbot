from src.query_expansion import expand_query


def test_expand_query_for_gender_terms():
    question = "Compare homens e mulheres"

    expanded = expand_query(question).lower()

    assert "sexo" in expanded
    assert "homem" in expanded
    assert "mulher" in expanded
    assert "rendimento" in expanded


def test_expand_query_for_race_color_terms():
    question = "As tabelas possuem dados por cor ou raça?"

    expanded = expand_query(question).lower()

    assert "cor ou raça" in expanded
    assert "branca" in expanded
    assert "preta" in expanded
    assert "parda" in expanded


def test_expand_query_for_income_values():
    question = "Quais são os valores de rendimento para Brasil e Norte?"

    expanded = expand_query(question).lower()

    assert "indicadores de rendimento do trabalho" in expanded
    assert "r$/mês" in expanded
    assert "brasil" in expanded
    assert "norte" in expanded


def test_expand_query_without_known_topic_returns_original_question():
    question = "Olá, tudo bem?"

    expanded = expand_query(question)

    assert expanded == question