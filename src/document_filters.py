"""
Document filtering utilities for retrieved RAG chunks.

This module contains lightweight filtering rules used after retrieval
or during metadata-based search decisions. The main goal is to reduce
noise in the final context sent to the LLM, especially when the user
asks for indicator values but the retriever also finds coefficient-of-
variation tables.

The functions are intentionally simple and transparent. They do not try
to replace semantic retrieval; instead, they encode domain-specific rules
that are easy to inspect and adjust.
"""


def user_wants_coefficient(question: str) -> bool:
    """
    Identify whether the user explicitly asks for coefficient-of-variation data.

    This is used to decide whether coefficient tables should be included
    in the retrieval step. If the user asks for coefficients, these tables
    should be preserved. Otherwise, they can be filtered out to reduce noise.

    Args:
        question: User question written in natural language.

    Returns:
        True if the question appears to request coefficient-of-variation
        information; False otherwise.
    """
    question_lower = question.lower()

    coefficient_terms = [
        "coeficiente",
        "coeficientes",
        "variação",
        "variacao",
        "cv",
    ]

    return any(term in question_lower for term in coefficient_terms)


def is_coefficient_document(document_text: str) -> bool:
    """
    Check whether a retrieved text chunk appears to come from a coefficient table.

    Args:
        document_text: Text content from a retrieved document or chunk.

    Returns:
        True if the text contains terms associated with coefficient-of-
        variation tables; False otherwise.
    """
    text_lower = document_text.lower()

    coefficient_document_terms = [
        "coeficientes de variação",
        "coeficientes de variacao",
    ]

    return any(term in text_lower for term in coefficient_document_terms)


def filter_retrieved_documents(question: str, docs):
    """
    Filter retrieved documents based on the user's apparent intent.

    If the user does not ask for coefficient-of-variation data, this function
    removes chunks that appear to come from coefficient tables. If all documents
    would be removed, the original list is returned to avoid an empty context.

    Args:
        question: Original user question.
        docs: List of retrieved LangChain Document objects.

    Returns:
        A filtered list of documents, or the original list if filtering would
        remove all available context.
    """
    if user_wants_coefficient(question):
        return docs

    filtered_docs = [
        doc for doc in docs
        if not is_coefficient_document(doc.page_content)
    ]

    if filtered_docs:
        return filtered_docs

    return docs