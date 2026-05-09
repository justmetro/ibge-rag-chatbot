"""
Embedding model configuration for the RAG pipeline.

This module centralizes the embedding model used by the project. Keeping
this configuration in a dedicated file makes it easier to replace the model,
compare alternatives and document the trade-offs behind the retrieval layer.

The current model, sentence-transformers/all-MiniLM-L6-v2, was chosen because
it is lightweight, free, runs locally and is fast enough for a Streamlit MVP.
"""

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Create and return the embedding model used by ChromaDB.

    The model runs locally through Sentence Transformers, avoiding API costs
    during the retrieval step. This is useful because only the final synthesis
    step depends on Gemini API; semantic search can run without paid services.

    Returns:
        HuggingFaceEmbeddings: LangChain-compatible embedding function.

    Notes:
        In local tests, retrieval using this embedding setup reached low
        latency for the current dataset. For larger datasets, model choice,
        chunk size and hardware should be reevaluated.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )