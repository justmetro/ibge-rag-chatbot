import os

from dotenv import load_dotenv
import google.genai as genai

from src.vectorstore import retrieve_documents
from src.query_expansion import expand_query
from src.document_filters import user_wants_coefficient
from src.document_filters import filter_retrieved_documents
from src.prompts import (
    GEMINI_SYSTEM_PROMPT,
    DEMO_FALLBACK_MESSAGE,
    GEMINI_ERROR_FALLBACK_MESSAGE,
)


load_dotenv()


class RAGBot:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.client = None

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

        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]

        last_error = None

        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                return response.text

            except Exception as e:
                last_error = e

        raise last_error

    def ask(self, question: str):
        expanded_question = expand_query(question)

        include_coefficients = user_wants_coefficient(question)

        docs = retrieve_documents(
            question=expanded_question,
            include_coefficients=include_coefficients,
        )

        if not docs:
            return {
                "answer": "Não encontrei informações suficientes nos documentos carregados.",
                "sources": [],
                "context_docs": [],
                "retrieved_context": "",
            }

        sources, trechos = self._formatar_trechos(docs)
        retrieved_context = "\n\n---\n\n".join(trechos)

        if self.client:
            try:
                resposta_gemini = self._responder_com_gemini(question, docs)

                return {
                    "answer": resposta_gemini,
                    "sources": sources,
                    "context_docs": docs,
                    "retrieved_context": retrieved_context,
                }

            except Exception as e:
                resumo = self._gerar_resumo_demo(docs)

                answer = (
                    f"{GEMINI_ERROR_FALLBACK_MESSAGE.strip()}\n\n"
                    f"Erro técnico: `{e}`\n\n"
                    f"{resumo}"
                )

                return {
                    "answer": answer,
                    "sources": sources,
                    "context_docs": docs,
                    "retrieved_context": retrieved_context,
                }

        resumo = self._gerar_resumo_demo(docs)

        answer = (
            f"{resumo}\n\n"
            f"> {DEMO_FALLBACK_MESSAGE.strip()}"
        )

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": docs,
            "retrieved_context": retrieved_context,
        }