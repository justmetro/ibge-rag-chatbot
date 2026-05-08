from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DATA_PATH = Path("data/documents.txt")
CHROMA_PATH = ".chroma"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_documents():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Arquivo data/documents.txt não encontrado.")

    text = DATA_PATH.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("O arquivo data/documents.txt está vazio.")

    return [
        Document(
            page_content=text,
            metadata={"source": "data/documents.txt"}
        )
    ]


def create_vectorstore():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def get_retriever():
    if not Path(CHROMA_PATH).exists():
        vectorstore = create_vectorstore()
    else:
        vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )