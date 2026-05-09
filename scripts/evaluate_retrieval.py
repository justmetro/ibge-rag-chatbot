"""
Retriever evaluation script for the IBGE RAG chatbot.

This script evaluates the retrieval layer using a small set of representative
questions. It computes heuristic retrieval metrics such as precision@k,
recall proxy, MRR and NDCG.

The evaluation is not a replacement for a fully labeled IR benchmark. It is a
lightweight validation tool designed to check whether the retriever returns
useful chunks for common user questions.
"""

import math
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.document_filters import user_wants_coefficient
from src.query_expansion import expand_query
from src.vectorstore import RETRIEVER_K, retrieve_documents


TEST_QUERIES = [
    {
        "question": "Quais indicadores aparecem nas tabelas?",
        "expected_terms": [
            "rendimento",
            "trabalho",
            "rendimento medio",
            "rendimento-hora",
        ],
    },
    {
        "question": "As tabelas tem dados de rendimento?",
        "expected_terms": [
            "rendimento",
            "r$/mes",
            "r$/hora",
            "trabalho principal",
        ],
    },
    {
        "question": "As tabelas possuem dados por sexo ou cor?",
        "expected_terms": [
            "sexo",
            "cor",
            "homem",
            "mulher",
            "branca",
        ],
    },
    {
        "question": "Quais sao os valores de rendimento medio real habitual do trabalho principal para Brasil, Norte e Rondonia?",
        "expected_terms": [
            "brasil",
            "norte",
            "rondonia",
            "rendimento medio real habitual",
        ],
    },
    {
        "question": "Compare o rendimento medio real habitual do trabalho principal entre homens e mulheres.",
        "expected_terms": [
            "homem",
            "mulher",
            "sexo",
            "rendimento medio real habitual",
        ],
    },
    {
        "question": "Mostre os coeficientes de variacao dos indicadores de rendimento.",
        "expected_terms": [
            "coeficientes de variacao",
            "rendimento",
            "brasil",
            "norte",
        ],
    },
]


def normalize_text(text: str) -> str:
    """
    Normalize text for simple keyword matching.

    Args:
        text: Input text.

    Returns:
        Lowercase text with common Portuguese accents removed.
    """
    replacements = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
        "ç": "c",
    }

    text = text.lower()

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def term_matches(text: str, terms: list[str]) -> list[str]:
    """
    Return expected terms found in the provided text.

    Args:
        text: Text to inspect.
        terms: Terms expected for a given query.

    Returns:
        List of matched terms.
    """
    normalized_text = normalize_text(text)

    return [
        term for term in terms
        if normalize_text(term) in normalized_text
    ]


def document_relevance_score(document_text: str, expected_terms: list[str]) -> float:
    """
    Estimate document relevance using matched expected terms.

    The score ranges from 0 to 1 and represents the fraction of expected terms
    found in a retrieved chunk.

    Args:
        document_text: Retrieved chunk text.
        expected_terms: Expected terms for the query.

    Returns:
        Relevance score between 0 and 1.
    """
    matched = term_matches(document_text, expected_terms)

    if not expected_terms:
        return 0.0

    return len(matched) / len(expected_terms)


def precision_at_k(relevance_scores: list[float], k: int) -> float:
    """
    Compute precision@k using non-zero relevance as relevant.

    Args:
        relevance_scores: Relevance scores for retrieved documents.
        k: Number of retrieved documents considered.

    Returns:
        Precision@k.
    """
    if k == 0:
        return 0.0

    relevant_count = sum(1 for score in relevance_scores[:k] if score > 0)

    return relevant_count / k


def recall_proxy(combined_text: str, expected_terms: list[str]) -> float:
    """
    Compute a proxy recall based on expected terms found in all retrieved chunks.

    Args:
        combined_text: Concatenated retrieved document text.
        expected_terms: Terms expected for the query.

    Returns:
        Fraction of expected terms found.
    """
    if not expected_terms:
        return 0.0

    matched = term_matches(combined_text, expected_terms)

    return len(matched) / len(expected_terms)


def reciprocal_rank(relevance_scores: list[float]) -> float:
    """
    Compute reciprocal rank for the first relevant result.

    Args:
        relevance_scores: Relevance scores for retrieved documents.

    Returns:
        Reciprocal rank. Returns 0 when no relevant result is found.
    """
    for index, score in enumerate(relevance_scores, start=1):
        if score > 0:
            return 1 / index

    return 0.0


def dcg(scores: list[float]) -> float:
    """
    Compute Discounted Cumulative Gain for graded relevance scores.

    Args:
        scores: Relevance scores.

    Returns:
        DCG value.
    """
    total = 0.0

    for index, score in enumerate(scores, start=1):
        total += score / math.log2(index + 1)

    return total


def ndcg(relevance_scores: list[float]) -> float:
    """
    Compute normalized DCG.

    Args:
        relevance_scores: Relevance scores for retrieved documents.

    Returns:
        NDCG score between 0 and 1.
    """
    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(question: str, expected_terms: list[str]) -> dict:
    """
    Evaluate retrieval quality for one query.

    Args:
        question: User-like test question.
        expected_terms: Terms expected to appear in useful retrieved chunks.

    Returns:
        Dictionary with retrieval metrics and diagnostic information.
    """
    expanded_question = expand_query(question)
    include_coefficients = user_wants_coefficient(question)

    start = time.perf_counter()

    docs = retrieve_documents(
        question=expanded_question,
        include_coefficients=include_coefficients,
    )

    end = time.perf_counter()

    elapsed = end - start

    document_texts = [
        doc.page_content
        for doc in docs
    ]

    combined_text = " ".join(document_texts)

    relevance_scores = [
        document_relevance_score(text, expected_terms)
        for text in document_texts
    ]

    matched_terms = term_matches(combined_text, expected_terms)

    k = len(docs)

    return {
        "question": question,
        "expanded_question": expanded_question,
        "docs_returned": k,
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
        "precision_at_k": round(precision_at_k(relevance_scores, k), 3),
        "recall_proxy": round(recall_proxy(combined_text, expected_terms), 3),
        "mrr": round(reciprocal_rank(relevance_scores), 3),
        "ndcg": round(ndcg(relevance_scores), 3),
        "latency_seconds": round(elapsed, 3),
        "passed": len(matched_terms) > 0,
    }


def print_result(result: dict) -> None:
    """
    Print one query evaluation result.

    Args:
        result: Evaluation result dictionary.
    """
    status = "PASSOU" if result["passed"] else "FALHOU"

    print(f"Pergunta: {result['question']}")
    print(f"Status: {status}")
    print(f"Documentos retornados: {result['docs_returned']}")
    print(f"Termos esperados: {', '.join(result['expected_terms'])}")
    print(f"Termos encontrados: {', '.join(result['matched_terms'])}")
    print(f"Precision@k: {result['precision_at_k']}")
    print(f"Recall proxy: {result['recall_proxy']}")
    print(f"MRR: {result['mrr']}")
    print(f"NDCG: {result['ndcg']}")
    print(f"Latencia: {result['latency_seconds']}s")
    print("-" * 80)


def save_report(results: list[dict], output_path: Path) -> None:
    """
    Save evaluation results to a text report.

    Args:
        results: List of query evaluation dictionaries.
        output_path: Destination report path.
    """
    total = len(results)
    passed = sum(1 for result in results if result["passed"])

    avg_precision = sum(result["precision_at_k"] for result in results) / total
    avg_recall = sum(result["recall_proxy"] for result in results) / total
    avg_mrr = sum(result["mrr"] for result in results) / total
    avg_ndcg = sum(result["ndcg"] for result in results) / total
    avg_latency = sum(result["latency_seconds"] for result in results) / total

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Avaliacao do Retriever RAG\n")
        file.write("=" * 80 + "\n\n")

        file.write("Observacao:\n")
        file.write(
            "Esta avaliacao usa metricas heuristicas baseadas em termos esperados. "
            "Ela nao substitui um benchmark rotulado formal, mas ajuda a monitorar "
            "qualidade de recuperacao e latencia durante o desenvolvimento.\n\n"
        )

        for result in results:
            status = "PASSOU" if result["passed"] else "FALHOU"

            file.write(f"Pergunta: {result['question']}\n")
            file.write(f"Status: {status}\n")
            file.write(f"Documentos retornados: {result['docs_returned']}\n")
            file.write(f"Termos esperados: {', '.join(result['expected_terms'])}\n")
            file.write(f"Termos encontrados: {', '.join(result['matched_terms'])}\n")
            file.write(f"Precision@k: {result['precision_at_k']}\n")
            file.write(f"Recall proxy: {result['recall_proxy']}\n")
            file.write(f"MRR: {result['mrr']}\n")
            file.write(f"NDCG: {result['ndcg']}\n")
            file.write(f"Latencia: {result['latency_seconds']}s\n")
            file.write("-" * 80 + "\n")

        file.write("\nResumo\n")
        file.write(f"Total de testes: {total}\n")
        file.write(f"Testes aprovados: {passed}\n")
        file.write(f"Taxa de sucesso simples: {passed / total:.0%}\n")
        file.write(f"Precision@k medio: {avg_precision:.3f}\n")
        file.write(f"Recall proxy medio: {avg_recall:.3f}\n")
        file.write(f"MRR medio: {avg_mrr:.3f}\n")
        file.write(f"NDCG medio: {avg_ndcg:.3f}\n")
        file.write(f"Latencia media: {avg_latency:.3f}s\n")


def main() -> None:
    """
    Run the full retrieval evaluation suite.
    """
    results = []

    print("\n=== Avaliacao do Retriever RAG ===\n")
    print(f"Retriever k configurado: {RETRIEVER_K}\n")

    for item in TEST_QUERIES:
        result = evaluate_query(
            question=item["question"],
            expected_terms=item["expected_terms"],
        )

        results.append(result)
        print_result(result)

    total = len(results)
    passed = sum(1 for result in results if result["passed"])

    avg_precision = sum(result["precision_at_k"] for result in results) / total
    avg_recall = sum(result["recall_proxy"] for result in results) / total
    avg_mrr = sum(result["mrr"] for result in results) / total
    avg_ndcg = sum(result["ndcg"] for result in results) / total
    avg_latency = sum(result["latency_seconds"] for result in results) / total

    print("\n=== Resumo ===")
    print(f"Total de testes: {total}")
    print(f"Testes aprovados: {passed}")
    print(f"Taxa de sucesso simples: {passed / total:.0%}")
    print(f"Precision@k medio: {avg_precision:.3f}")
    print(f"Recall proxy medio: {avg_recall:.3f}")
    print(f"MRR medio: {avg_mrr:.3f}")
    print(f"NDCG medio: {avg_ndcg:.3f}")
    print(f"Latencia media: {avg_latency:.3f}s")

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "retrieval_evaluation.txt"

    save_report(results, output_path)

    print(f"\nRelatorio salvo em: {output_path}")


if __name__ == "__main__":
    main()