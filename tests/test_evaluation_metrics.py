from scripts.evaluate_retrieval import (
    document_relevance_score,
    ndcg,
    precision_at_k,
    recall_proxy,
    reciprocal_rank,
    term_matches,
)


def test_term_matches_finds_expected_terms():
    text = "Brasil possui rendimento medio real habitual do trabalho principal."
    terms = ["brasil", "rendimento medio", "trabalho"]

    matches = term_matches(text, terms)

    assert matches == terms


def test_document_relevance_score_partial_match():
    text = "Brasil e Norte aparecem na tabela."
    terms = ["brasil", "norte", "rondonia"]

    score = document_relevance_score(text, terms)

    assert round(score, 3) == 0.667


def test_precision_at_k_counts_relevant_documents():
    scores = [1.0, 0.5, 0.0, 0.0]

    result = precision_at_k(scores, 4)

    assert result == 0.5


def test_recall_proxy_counts_expected_terms_found():
    combined_text = "Brasil Norte Rondonia rendimento medio"
    terms = ["brasil", "norte", "rondonia", "rendimento medio"]

    result = recall_proxy(combined_text, terms)

    assert result == 1.0


def test_reciprocal_rank_first_relevant_document():
    scores = [0.0, 0.5, 1.0]

    result = reciprocal_rank(scores)

    assert result == 0.5


def test_ndcg_is_one_for_ideal_ordering():
    scores = [1.0, 0.5, 0.0]

    result = ndcg(scores)

    assert result == 1.0