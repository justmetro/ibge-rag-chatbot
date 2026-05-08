from src.vectorstore import get_retriever


class RAGBot:
    def __init__(self):
        self.retriever = get_retriever()

    def _gerar_resumo_demo(self, docs):
        texto_completo = " ".join(
            doc.page_content.lower()
            for doc in docs
        )

        achados = []

        if "rendimento médio" in texto_completo:
            achados.append("rendimento médio real habitual")

        if "rendimento-hora" in texto_completo:
            achados.append("rendimento-hora médio real habitual")

        if "trabalho principal" in texto_completo:
            achados.append("rendimento do trabalho principal")

        if "todos os trabalhos" in texto_completo:
            achados.append("rendimento de todos os trabalhos")

        if "sexo" in texto_completo:
            achados.append("recortes por sexo")

        if "cor ou raça" in texto_completo:
            achados.append("recortes por cor ou raça")

        if "grandes regiões" in texto_completo or "unidades da federação" in texto_completo:
            achados.append("recortes territoriais por Grandes Regiões e Unidades da Federação")

        if not achados:
            return (
                "Com base nos trechos recuperados, foram encontrados dados relevantes "
                "nas tabelas carregadas, mas sem uma categoria específica identificada automaticamente."
            )

        achados_unicos = []
        for item in achados:
            if item not in achados_unicos:
                achados_unicos.append(item)

        return (
            "Com base nos trechos recuperados, as tabelas carregadas parecem conter indicadores relacionados a "
            + ", ".join(achados_unicos)
            + "."
        )

    def ask(self, question: str):
        docs = self.retriever.invoke(question)

        if not docs:
            return {
                "answer": "Não encontrei informações suficientes nos documentos carregados.",
                "sources": [],
                "context_docs": []
            }

        sources = []
        trechos = []

        resumo = self._gerar_resumo_demo(docs)

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Fonte desconhecida")
            trecho = doc.page_content.strip()
            trecho_formatado = trecho.replace("```", "")

            if source not in sources:
                sources.append(source)

            bloco = (
                f"### Trecho {i}\n\n"
                f"**Fonte:** `{source}`\n\n"
                "```text\n"
                f"{trecho_formatado}\n"
                "```"
            )

            trechos.append(bloco)

        answer = (
            f"{resumo}\n\n"
            "Abaixo estão os trechos mais relevantes encontrados pelo sistema de busca semântica:\n\n"
            + "\n\n---\n\n".join(trechos)
            + "\n\n> Modo demo: por enquanto o app retorna uma síntese simples e os trechos recuperados pelo RAG. "
            "Quando uma API de LLM estiver ativa, esses trechos serão usados para gerar uma resposta mais natural e contextualizada."
        )

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": docs
        }