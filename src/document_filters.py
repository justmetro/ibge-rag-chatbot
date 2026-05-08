def user_wants_coefficient(question: str) -> bool:
    """
    Identifica se a pergunta do usuário pede explicitamente coeficientes de variação.

    O objetivo é evitar remover tabelas de coeficiente quando elas são exatamente
    o que o usuário quer consultar.
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
    Verifica se um trecho recuperado parece vir de uma tabela de coeficientes de variação.
    """
    text_lower = document_text.lower()

    coefficient_document_terms = [
        "coeficientes de variação",
        "coeficientes de variacao",
    ]

    return any(term in text_lower for term in coefficient_document_terms)


def filter_retrieved_documents(question: str, docs):
    """
    Filtra documentos recuperados pelo retriever de acordo com a intenção da pergunta.

    Se o usuário não pedir coeficientes de variação, removemos trechos dessas tabelas
    para reduzir ruído na resposta final.

    Se todos os documentos forem removidos pelo filtro, retornamos os documentos
    originais para evitar resposta vazia.
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