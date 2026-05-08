from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.embedder import get_embeddings


DATA_PATH = Path("data/documents.txt")
CHROMA_PATH = ".chroma"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVER_K = 5


def load_documents():
    """
    Carrega o arquivo textual gerado a partir das tabelas do IBGE.

    O arquivo data/documents.txt é produzido pelo table_loader.py.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError("Arquivo data/documents.txt não encontrado.")

    text = DATA_PATH.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("O arquivo data/documents.txt está vazio.")

    return [
        Document(
            page_content=text,
            metadata={"source": str(DATA_PATH)}
        )
    ]


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


def get_retriever():
    """
    Retorna o retriever usado pelo RAG.

    Caso a base Chroma ainda não exista, ela é criada automaticamente.
    """
    if not Path(CHROMA_PATH).exists():
        vectorstore = create_vectorstore()
    else:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )