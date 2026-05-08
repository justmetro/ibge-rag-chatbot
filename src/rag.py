import os

from dotenv import load_dotenv
import google.genai as genai

from src.query_expansion import expand_query
from src.vectorstore import get_retriever
from src.prompts import (
    GEMINI_SYSTEM_PROMPT,
    DEMO_FALLBACK_MESSAGE,
    GEMINI_ERROR_FALLBACK_MESSAGE,
)


load_dotenv()


class RAGBot:
    def __init__(self):
        self.retriever = get_retriever()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.client = None

    def _filtrar_docs_por_pergunta(self, question, docs):
        pergunta = question.lower()

        usuario_quer_coeficiente = (
            "coeficiente" in pergunta
            or "variação" in pergunta
            or "variacao" in pergunta
            or "cv" in pergunta
        )

        if usuario_quer_coeficiente:
            return docs

        docs_sem_coeficiente = [
            doc
            for doc in docs
            if "coeficientes de variação" not in doc.page_content.lower()
            and "coeficientes de variacao" not in doc.page_content.lower()
        ]

        if docs_sem_coeficiente:
            return docs_sem_coeficiente

        return docs

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

        if (
            "grandes regiões" in texto_completo
            or "unidades da federação" in texto_completo
        ):
            achados.append(
                "recortes territoriais por Grandes Regiões e Unidades da Federação"
            )

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

    def _formatar_trechos(self, docs):
        sources = []
        trechos = []

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

        return sources, trechos

    def _responder_com_gemini(self, question, docs):
        contexto = "\n\n".join(
            [
                f"Fonte: {doc.metadata.get('source', 'Fonte desconhecida')}\n{doc.page_content}"
                for doc in docs
            ]
        )

        prompt = GEMINI_SYSTEM_PROMPT.format(
            question=question,
            context=contexto,
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    def ask(self, question: str):
        expanded_question = expand_query(question)
        docs = self.retriever.invoke(expanded_question)
        docs = self._filtrar_docs_por_pergunta(question, docs)

        if not docs:
            return {
                "answer": "Não encontrei informações suficientes nos documentos carregados.",
                "sources": [],
                "context_docs": [],
            }

        sources, trechos = self._formatar_trechos(docs)

        if self.client:
            try:
                resposta_gemini = self._responder_com_gemini(question, docs)

                answer = (
                    f"{resposta_gemini}\n\n"
                    "---\n\n"
                    "### Trechos utilizados pelo RAG\n\n"
                    + "\n\n---\n\n".join(trechos)
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "context_docs": docs,
                }

            except Exception as e:
                resumo = self._gerar_resumo_demo(docs)

                answer = (
                    f"{GEMINI_ERROR_FALLBACK_MESSAGE.strip()}\n\n"
                    f"Erro técnico: `{e}`\n\n"
                    f"{resumo}\n\n"
                    "Abaixo estão os trechos mais relevantes encontrados pelo sistema de busca semântica:\n\n"
                    + "\n\n---\n\n".join(trechos)
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "context_docs": docs,
                }

        resumo = self._gerar_resumo_demo(docs)

        answer = (
            f"{resumo}\n\n"
            "Abaixo estão os trechos mais relevantes encontrados pelo sistema de busca semântica:\n\n"
            + "\n\n---\n\n".join(trechos)
            + f"\n\n> {DEMO_FALLBACK_MESSAGE.strip()}"
        )

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": docs,
        }