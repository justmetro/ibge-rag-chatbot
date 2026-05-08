from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Cria e retorna o modelo de embeddings usado pelo projeto.

    O modelo all-MiniLM-L6-v2 é leve, gratuito e roda localmente,
    sendo adequado para protótipos RAG com baixo custo.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )