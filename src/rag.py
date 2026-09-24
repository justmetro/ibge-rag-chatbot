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

import logging
import os

import google.genai as genai
from google.genai import errors, types

from src.config import GEMINI_MODELS
from src.vectorstore import format_source_label, retrieve_documents
from src.query_expansion import expand_query
from src.text_utils import normalize_text
from src.document_filters import user_wants_coefficient
from src.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    DEMO_FALLBACK_MESSAGE,
    LLM_ERROR_FALLBACK_MESSAGE,
)


logger = logging.getLogger(__name__)
ANSWER_CACHE_SIZE = 200


def is_api_key_error(error: errors.APIError) -> bool:
    """
    Check whether a Gemini API error is caused by a missing or invalid key.

    Args:
        error: Error raised by the google-genai client.

    Returns:
        True for authentication/permission failures; False otherwise.
    """
    if error.code in (401, 403):
        return True

    return error.code == 400 and "api key" in str(error).lower()


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

        # Answers already generated, keyed by normalized question. Repeated
        # questions (e.g. the suggested ones) do not consume free-tier quota.
        self._respostas = {}

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
            source = format_source_label(doc.metadata)
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

        The models listed in GEMINI_MODELS are tried in order. On the free
        tier each model has its own quota, so when one model hits its limit
        (HTTP 429) or is unavailable, the next one is tried. Invalid API key
        errors stop immediately, since every model would fail the same way.

        Args:
            question: Original user question.
            docs: Retrieved documents used as context.

        Returns:
            Markdown answer generated by Gemini.

        Raises:
            Exception: The last error raised if every configured model fails.
        """
        contexto = "\n\n".join(
            [
                f"Fonte: {format_source_label(doc.metadata)}\n{doc.page_content}"
                for doc in docs
            ]
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            context=contexto,
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

        last_error = None

        for model_name in GEMINI_MODELS:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )

                if response.text and response.text.strip():
                    return response.text.strip()

                last_error = ValueError(f"Resposta vazia do modelo {model_name}.")

            except errors.APIError as error:
                logger.warning("Falha no modelo Gemini %s: %s", model_name, error)
                last_error = error

                if is_api_key_error(error):
                    break

        raise last_error or ValueError("Nenhum modelo configurado em GEMINI_MODELS.")

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

        cache_key = normalize_text(question).strip()

        if self.client and cache_key in self._respostas:
            return {
                "answer": self._respostas[cache_key],
                "sources": sources,
                "context_docs": docs,
                "retrieved_context": retrieved_context,
            }

        if self.client:
            try:
                resposta = self._responder_com_gemini(question, docs)

                if len(self._respostas) >= ANSWER_CACHE_SIZE:
                    self._respostas.pop(next(iter(self._respostas)))

                self._respostas[cache_key] = resposta

                return {
                    "answer": resposta,
                    "sources": sources,
                    "context_docs": docs,
                    "retrieved_context": retrieved_context,
                }

            except Exception as error:
                logger.error("Gemini indisponível; usando modo demo: %s", error)
                resumo = self._gerar_resumo_demo(docs)

                answer = f"{LLM_ERROR_FALLBACK_MESSAGE.strip()}\n\n{resumo}"

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