import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.vectorstore import get_retriever
from src.query_expansion import expand_query


TEST_QUERIES = [
    {
        "question": "Quais indicadores aparecem nas tabelas?",
        "expected_terms": ["rendimento", "trabalho"],
    },
    {
        "question": "As tabelas tem dados de rendimento?",
        "expected_terms": ["rendimento", "r$/mes"],
    },
    {
        "question": "As tabelas possuem dados por sexo ou cor?",
        "expected_terms": ["sexo", "cor"],
    },
    {
        "question": "Quais sao os valores de rendimento medio real habitual do trabalho principal para Brasil, Norte e Rondonia?",
        "expected_terms": ["brasil", "norte", "rond"],
    },
    {
        "question": "Compare o rendimento medio real habitual do trabalho principal entre homens e mulheres.",
        "expected_terms": ["homem", "mulher"],
    },
]


def normalize_text(text):
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


def evaluate_query(retriever, question, expected_terms):
    start = time.perf_counter()

    expanded_question = expand_query(question)
    docs = retriever.invoke(expanded_question)

    end = time.perf_counter()

    elapsed = end - start

    combined_text = " ".join(
        normalize_text(doc.page_content)
        for doc in docs
    )

    matched_terms = [
        term for term in expected_terms
        if normalize_text(term) in combined_text
    ]

    return {
        "question": question,
        "docs_returned": len(docs),
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
        "latency_seconds": round(elapsed, 3),
        "passed": len(matched_terms) > 0,
    }


def main():
    retriever = get_retriever()

    results = []

    print("\n=== Avaliacao simples do Retriever RAG ===\n")

    for item in TEST_QUERIES:
        result = evaluate_query(
            retriever=retriever,
            question=item["question"],
            expected_terms=item["expected_terms"],
        )

        results.append(result)

        status = "PASSOU" if result["passed"] else "FALHOU"

        print(f"Pergunta: {result['question']}")
        print(f"Status: {status}")
        print(f"Documentos retornados: {result['docs_returned']}")
        print(f"Termos esperados: {', '.join(result['expected_terms'])}")
        print(f"Termos encontrados: {', '.join(result['matched_terms'])}")
        print(f"Latencia: {result['latency_seconds']}s")
        print("-" * 80)

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    avg_latency = sum(result["latency_seconds"] for result in results) / total

    print("\n=== Resumo ===")
    print(f"Total de testes: {total}")
    print(f"Testes aprovados: {passed}")
    print(f"Taxa de sucesso simples: {passed / total:.0%}")
    print(f"Latencia media: {avg_latency:.3f}s")

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "retrieval_evaluation.txt"

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Avaliacao simples do Retriever RAG\n")
        file.write("=" * 80 + "\n\n")

        for result in results:
            status = "PASSOU" if result["passed"] else "FALHOU"

            file.write(f"Pergunta: {result['question']}\n")
            file.write(f"Status: {status}\n")
            file.write(f"Documentos retornados: {result['docs_returned']}\n")
            file.write(f"Termos esperados: {', '.join(result['expected_terms'])}\n")
            file.write(f"Termos encontrados: {', '.join(result['matched_terms'])}\n")
            file.write(f"Latencia: {result['latency_seconds']}s\n")
            file.write("-" * 80 + "\n")

        file.write("\nResumo\n")
        file.write(f"Total de testes: {total}\n")
        file.write(f"Testes aprovados: {passed}\n")
        file.write(f"Taxa de sucesso simples: {passed / total:.0%}\n")
        file.write(f"Latencia media: {avg_latency:.3f}s\n")

    print(f"\nRelatorio salvo em: {output_path}")


if __name__ == "__main__":
    main()