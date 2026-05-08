import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.embedder import get_embeddings


DATA_PATH = Path("data/documents.txt")
CHROMA_PATH = ".chroma"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "12"))


def is_coefficient_text(text: str) -> bool:
    text_lower = text.lower()

    return (
        "coeficientes de variação" in text_lower
        or "coeficientes de variacao" in text_lower
    )


def split_text_into_blocks(text: str):
    """
    Divide o documents.txt em blocos menores antes do chunking.

    O table_loader.py separa tabelas com linhas de '='.
    Separar por blocos ajuda o Chroma a recuperar tabelas mais específicas.
    """
    raw_blocks = text.split("=" * 80)

    blocks = [
        block.strip()
        for block in raw_blocks
        if block.strip()
    ]

    if blocks:
        return blocks

    return [text]


def load_documents():
    """
    Carrega o arquivo textual gerado a partir das tabelas do IBGE.

    Cada bloco recebe metadata indicando se é tabela de coeficiente de variação.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError("Arquivo data/documents.txt não encontrado.")

    text = DATA_PATH.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("O arquivo data/documents.txt está vazio.")

    blocks = split_text_into_blocks(text)

    documents = []

    for index, block in enumerate(blocks, start=1):
        documents.append(
            Document(
                page_content=block,
                metadata={
                    "source": str(DATA_PATH),
                    "block_id": index,
                    "is_coefficient": is_coefficient_text(block),
                },
            )
        )

    return documents


def split_documents(documents):
    """
    Divide documentos longos em chunks menores para recuperação semântica.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


def create_vectorstore():
    """
    Cria a base vetorial Chroma a partir dos documentos processados.
    """
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


def load_vectorstore():
    """
    Carrega uma base vetorial Chroma já existente.
    """
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def retrieve_documents(question: str, include_coefficients: bool = False):
    """
    Recupera documentos relevantes no ChromaDB.

    Se include_coefficients=False, remove tabelas de coeficientes diretamente
    na busca usando metadata.
    """
    if not Path(CHROMA_PATH).exists():
        vectorstore = create_vectorstore()
    else:
        vectorstore = load_vectorstore()

    if include_coefficients:
        return vectorstore.similarity_search(
            query=question,
            k=RETRIEVER_K,
        )

    return vectorstore.similarity_search(
        query=question,
        k=RETRIEVER_K,
        filter={"is_coefficient": False},
    )


def get_retriever():
    """
    Mantido por compatibilidade com scripts antigos.
    """
    if not Path(CHROMA_PATH).exists():
        vectorstore = create_vectorstore()
    else:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )