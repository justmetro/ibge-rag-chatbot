"""
RAG orchestration module.

This module contains the RAGBot class, which coordinates the complete retrieval-
augmented generation flow:

1. Expand the user query with domain-specific terms.
2. Retrieve relevant table chunks from ChromaDB.
3. Optionally filter documents using metadata and intent rules.
4. Send the recovered context to Gemini API when available.
5. Fall back to a local demo response when the API is unavailable.

The class is designed to keep the Streamlit app simple. The UI calls only
RAGBot.ask(), while this module handles retrieval, prompt construction,
generation and fallback behavior.
"""

import os

from dotenv import load_dotenv
import google.genai as genai

from src.vectorstore import retrieve_documents
from src.query_expansion import expand_query
from src.document_filters import user_wants_coefficient
from src.prompts import (
    GEMINI_SYSTEM_PROMPT,
    DEMO_FALLBACK_MESSAGE,
    GEMINI_ERROR_FALLBACK_MESSAGE,
)


load_dotenv()


class RAGBot:
    """
    Retrieval-Augmented Generation bot for IBGE social indicators.

    The bot retrieves relevant chunks from a ChromaDB vector store and uses
    Gemini API to synthesize a natural-language answer when an API key is
    available. If the API fails or is not configured, it returns a deterministic
    demo response based on retrieved context.
    """

    def __init__(self):
        """
        Initialize the RAG bot and configure the Gemini client when available.

        The Gemini API key is loaded from the GEMINI_API_KEY environment
        variable. If the key is absent, the bot remains usable through the
        fallback demo mode.
        """
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.client = None

    def _gerar_resumo_demo(self, docs):
        """
        Generate a simple deterministic summary from retrieved documents.

        This method is used when Gemini is not available. It does not attempt
        to produce a full answer; instead, it detects recurring terms in the
        retrieved chunks and produces a short description of the topics found.

        Args:
            docs: List of retrieved LangChain Document objects.

        Returns:
            A short Portuguese summary based on detected terms.
        """
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
        """
        Format retrieved documents for display in the Streamlit expander.

        Args:
            docs: List of retrieved LangChain Document objects.

        Returns:
            A tuple with:
                - sources: list of unique source names.
                - trechos: list of Markdown-formatted retrieved chunks.
        """
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
        """
        Generate an answer using Gemini API and retrieved RAG context.

        Multiple Gemini models are tried in sequence to reduce temporary
        failures caused by high demand or availability issues. If all models
        fail, the last error is raised and handled by ask().

        Args:
            question: Original user question.
            docs: Retrieved documents used as context.

        Returns:
            Text answer generated by Gemini.

        Raises:
            Exception: The last exception raised by the Gemini client if all
            configured models fail.
        """
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
        ]

        last_error = None

        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                return response.text

            except Exception as error:
                last_error = error

        raise last_error

    def ask(self, question: str):
        """
        Answer a user question using the RAG pipeline.

        The method expands the query, decides whether coefficient tables should
        be included, retrieves relevant chunks, formats context for the UI and
        attempts to generate a Gemini answer. If Gemini is unavailable, it
        returns a fallback response and still exposes retrieved context.

        Args:
            question: User question submitted through the Streamlit interface.

        Returns:
            Dictionary with:
                - answer: final text shown to the user.
                - sources: list of sources used.
                - context_docs: retrieved LangChain Document objects.
                - retrieved_context: Markdown-formatted context for display.
        """
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

            except Exception:
                resumo = self._gerar_resumo_demo(docs)

                answer = (
                    "A API generativa não respondeu no momento, então o app usou "
                    "o modo demo com os trechos recuperados pelo RAG.\n\n"
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