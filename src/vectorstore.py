"""
Vector store creation and retrieval logic.

This module is responsible for transforming the processed IBGE text files
into LangChain Document objects, splitting them into chunks, embedding them
and storing them in ChromaDB.

It also exposes a retrieval function that supports metadata filtering. This
is used to avoid coefficient-of-variation tables when the user asks for
actual indicator values.

The persisted index stores a fingerprint of its inputs (source files, chunk
settings and embedding model). When any of them changes, or a previous build
was interrupted, the index is rebuilt automatically instead of silently
serving stale data.
"""

import hashlib
import re
import shutil
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PATH, DOCUMENTS_PATH, PROCESSED_PDFS_PATH, RETRIEVER_K
from src.document_filters import is_coefficient_document
from src.embedder import EMBEDDING_MODEL_NAME, get_embeddings


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_TAIL_CHUNK_SIZE = 300

# Bump when the chunk text or metadata format changes, to force a rebuild.
INDEX_FORMAT_VERSION = 3

FINGERPRINT_PATH = CHROMA_PATH / "index_fingerprint.txt"


def is_coefficient_text(text: str) -> bool:
    """
    Check whether a text block comes from a coefficient-of-variation table.

    Args:
        text: Text block extracted from the processed IBGE documents.

    Returns:
        True if the block appears to represent coefficient-of-variation data;
        False otherwise.
    """
    return is_coefficient_document(text)


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


def parse_block_header(block: str) -> dict:
    """
    Extract source file, sheet, page and table title from a block header.

    Blocks written by the table and PDF loaders start with lines such as
    "Fonte: Tabela 1.1 (Indic_BR).xls" and "Aba: 2024" (or "Página: 3").
    IBGE spreadsheets also carry a descriptive title row, such as
    "- Tabela 1.1 - Indicadores estruturais do mercado de trabalho ...".

    Args:
        block: One table-level text block.

    Returns:
        Dictionary with the keys found among "source", "sheet", "page" and
        "title".
    """
    patterns = {
        "source": r"^Fonte:\s*(.+)$",
        "sheet": r"^Aba:\s*(.+)$",
        "page": r"^Página:\s*(.+)$",
        "title": r"^- (Tabela .+)$",
    }

    header = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, block, flags=re.MULTILINE)

        if match:
            header[key] = match.group(1).strip()

    return header


def format_source_label(metadata: dict) -> str:
    """
    Build a human-readable source label from document metadata.

    Args:
        metadata: Document metadata with "source" and optional "sheet"/"page".

    Returns:
        Label such as "Tabela 1.1 (Indic_BR).xls (aba 2024)".
    """
    label = metadata.get("source", "Fonte desconhecida")

    if metadata.get("sheet"):
        label += f" (aba {metadata['sheet']})"
    elif metadata.get("page"):
        label += f" (página {metadata['page']})"

    return label


def get_source_paths():
    """
    List the processed text files that feed the vector store.

    Returns:
        Existing paths among data/documents.txt and data/processed_pdfs.txt.
    """
    return [
        path
        for path in (DOCUMENTS_PATH, PROCESSED_PDFS_PATH)
        if path.exists()
    ]


def load_documents():
    """
    Load processed IBGE text data as LangChain Document objects.

    Each block receives metadata indicating its original source file, sheet
    or page, block ID and whether it represents coefficient-of-variation
    data. The coefficient flag is later used by ChromaDB filters.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If data/documents.txt does not exist.
        ValueError: If the processed files are empty.
    """
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            "Arquivo data/documents.txt não encontrado. "
            "Rode: python src/table_loader.py"
        )

    documents = []

    for path in get_source_paths():
        text = path.read_text(encoding="utf-8")

        for block in split_text_into_blocks(text):
            if not block.strip():
                continue

            header = parse_block_header(block)

            metadata = {
                "source": header.get("source", path.name),
                "block_id": len(documents) + 1,
                "is_coefficient": is_coefficient_text(block),
            }

            if "sheet" in header:
                metadata["sheet"] = header["sheet"]

            if "page" in header:
                metadata["page"] = header["page"]

            if "title" in header:
                metadata["title"] = header["title"]

            documents.append(Document(page_content=block, metadata=metadata))

    if not documents:
        raise ValueError("Os arquivos processados estão vazios.")

    return documents


def split_documents(documents):
    """
    Split loaded documents into smaller chunks for semantic retrieval.

    A short last chunk (usually only table footnotes) is merged into the
    previous one. On its own it would carry little data but still match
    many questions, crowding out chunks with actual values.

    Only the first chunk of a block contains the "Fonte:" header, so the
    table title (or the source label when there is no title) is prepended to
    the remaining chunks. This keeps table identity (e.g. "segundo sexo e cor
    ou raça") visible to the embedding model.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked LangChain Document objects.
    """
    # Blocks are line-oriented (one table row per line). Splitting on blank
    # lines would isolate the "Fonte:/Aba:" header as a chunk of its own.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", " ", ""],
        add_start_index=True,
    )

    chunks = []

    for document in documents:
        document_chunks = splitter.split_documents([document])

        if (
            len(document_chunks) > 1
            and len(document_chunks[-1].page_content) < MIN_TAIL_CHUNK_SIZE
        ):
            previous = document_chunks[-2]
            start = previous.metadata["start_index"]
            previous.page_content = document.page_content[start:].strip()
            document_chunks.pop()

        chunks.extend(document_chunks)

    for chunk in chunks:
        if not chunk.page_content.startswith("Fonte:"):
            label = chunk.metadata.get("title") or format_source_label(chunk.metadata)
            chunk.page_content = f"[{label}]\n{chunk.page_content}"

    return chunks


def compute_fingerprint() -> str:
    """
    Hash every input that affects the index contents.

    Returns:
        Hex digest combining source files, chunk settings, index format and
        embedding model.
    """
    digest = hashlib.sha256()
    digest.update(
        f"{INDEX_FORMAT_VERSION}|{EMBEDDING_MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode()
    )

    for path in get_source_paths():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())

    return digest.hexdigest()


def index_is_current() -> bool:
    """
    Check whether the persisted index was fully built from the current inputs.

    The fingerprint file is written only after a successful build, so an
    interrupted build is also detected as outdated.

    Returns:
        True if the index can be reused; False if it must be rebuilt.
    """
    if not FINGERPRINT_PATH.exists():
        return False

    return FINGERPRINT_PATH.read_text(encoding="utf-8").strip() == compute_fingerprint()


def create_vectorstore():
    """
    Create and persist a ChromaDB vector store from processed documents.

    Any previous index is removed first, so documents are never duplicated.

    Returns:
        Chroma: Persisted Chroma vector store instance.
    """
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    FINGERPRINT_PATH.write_text(compute_fingerprint(), encoding="utf-8")

    return vectorstore


def load_vectorstore():
    """
    Load an existing ChromaDB vector store from disk.

    Returns:
        Chroma: Previously persisted vector store instance.
    """
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    """
    Return the vector store, building it only when missing or outdated.

    The instance is cached for the whole process, so the embedding model and
    the ChromaDB client are loaded once instead of on every question.

    Returns:
        Chroma: Ready-to-query vector store.
    """
    if index_is_current():
        return load_vectorstore()

    return create_vectorstore()


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
    vectorstore = get_vectorstore()

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
    return get_vectorstore().as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )


if __name__ == "__main__":
    create_vectorstore()
    print(f"Índice criado com sucesso em: {CHROMA_PATH}")
