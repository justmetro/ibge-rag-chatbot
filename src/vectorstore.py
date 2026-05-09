"""
Vector store creation and retrieval logic.

This module is responsible for transforming the processed IBGE text file
into LangChain Document objects, splitting them into chunks, embedding them
and storing them in ChromaDB.

It also exposes a retrieval function that supports metadata filtering. This
is used to avoid coefficient-of-variation tables when the user asks for
actual indicator values.
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embedder import get_embeddings


DATA_PATH = Path("data/documents.txt")
CHROMA_PATH = ".chroma"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))


def is_coefficient_text(text: str) -> bool:
    """
    Check whether a text block comes from a coefficient-of-variation table.

    Args:
        text: Text block extracted from the processed IBGE documents.

    Returns:
        True if the block appears to represent coefficient-of-variation data;
        False otherwise.
    """
    text_lower = text.lower()

    return (
        "coeficientes de variação" in text_lower
        or "coeficientes de variacao" in text_lower
    )


def split_text_into_blocks(text: str):
    """
    Split the processed text file into table-level blocks.

    The table loader separates extracted tables using lines of '=' characters.
    Splitting by these separators before chunking helps preserve table context
    and allows metadata to be assigned at the block level.

    Args:
        text: Full text content from data/documents.txt.

    Returns:
        List of non-empty text blocks. If no separator is found, returns the
        full text as a single block.
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
    Load processed IBGE text data as LangChain Document objects.

    Each block receives metadata indicating its source, block ID and whether
    it represents coefficient-of-variation data. This metadata is later used
    by ChromaDB filters during retrieval.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If data/documents.txt does not exist.
        ValueError: If data/documents.txt exists but is empty.
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
    Split loaded documents into smaller chunks for semantic retrieval.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked LangChain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def create_vectorstore():
    """
    Create and persist a ChromaDB vector store from processed documents.

    Returns:
        Chroma: Persisted Chroma vector store instance.
    """
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    return vectorstore


def load_vectorstore():
    """
    Load an existing ChromaDB vector store from disk.

    Returns:
        Chroma: Previously persisted vector store instance.
    """
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )


def retrieve_documents(question: str, include_coefficients: bool = False):
    """
    Retrieve relevant chunks from ChromaDB.

    When include_coefficients is False, coefficient-of-variation chunks are
    filtered out using metadata. This helps reduce noise when the user asks
    for actual indicator values rather than statistical precision measures.

    Args:
        question: Expanded or original user query.
        include_coefficients: Whether coefficient-of-variation documents should
            be allowed in the search results.

    Returns:
        List of relevant LangChain Document objects.
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
    Return a LangChain retriever for compatibility with older scripts.

    New code should prefer retrieve_documents(), because it supports metadata
    filtering. This function is kept to avoid breaking previous integrations.

    Returns:
        LangChain retriever object backed by ChromaDB.
    """
    if not Path(CHROMA_PATH).exists():
        vectorstore = create_vectorstore()
    else:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )