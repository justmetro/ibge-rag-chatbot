from src.text_utils import contains_any_term, normalize_text


def test_normalize_text_removes_accents_and_lowercases():
    assert normalize_text("Rondônia, Região e Ação") == "rondonia, regiao e acao"


def test_contains_any_term_matches_whole_words_only():
    assert contains_any_term("Quais recortes aparecem?", ["cor"]) is False
    assert contains_any_term("Dados por cor ou raça", ["cor"]) is True


def test_contains_any_term_matches_phrases_and_symbols():
    assert contains_any_term("Valores em R$ por mês", ["r$"]) is True
    assert contains_any_term("Rendimento médio real", ["rendimento medio"]) is True
