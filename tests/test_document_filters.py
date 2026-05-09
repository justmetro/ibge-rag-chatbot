from langchain_core.documents import Document

from src.document_filters import (
    filter_retrieved_documents,
    is_coefficient_document,
    user_wants_coefficient,
)


def test_user_wants_coefficient_when_question_mentions_coeficiente():
    question = "Mostre os coeficientes de variação dos indicadores"

    assert user_wants_coefficient(question) is True


def test_user_does_not_want_coefficient_for_income_values():
    question = "Quais são os valores de rendimento médio?"

    assert user_wants_coefficient(question) is False


def test_is_coefficient_document_detects_coefficient_table():
    text = "Tabela 1.5 - Coeficientes de variação dos indicadores de rendimento"

    assert is_coefficient_document(text) is True


def test_is_coefficient_document_ignores_regular_indicator_table():
    text = "Tabela 1.5 - Indicadores de rendimento do trabalho"

    assert is_coefficient_document(text) is False


def test_filter_retrieved_documents_removes_coefficients_when_not_requested():
    docs = [
        Document(page_content="Tabela - Indicadores de rendimento do trabalho"),
        Document(page_content="Tabela - Coeficientes de variação dos indicadores"),
    ]

    filtered = filter_retrieved_documents(
        question="Quais são os valores de rendimento?",
        docs=docs,
    )

    assert len(filtered) == 1
    assert "Indicadores de rendimento" in filtered[0].page_content


def test_filter_retrieved_documents_keeps_coefficients_when_requested():
    docs = [
        Document(page_content="Tabela - Indicadores de rendimento do trabalho"),
        Document(page_content="Tabela - Coeficientes de variação dos indicadores"),
    ]

    filtered = filter_retrieved_documents(
        question="Mostre os coeficientes de variação",
        docs=docs,
    )

    assert len(filtered) == 2