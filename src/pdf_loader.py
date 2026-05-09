"""
PDF loading utilities for optional IBGE document ingestion.

This module extracts text from PDF files stored in data/pdfs and writes the
result to data/processed_pdfs.txt. It is not required for the current table-based
MVP, but it makes the project extensible to IBGE reports, methodological notes
and other public documents distributed as PDFs.

Example:
    python src/pdf_loader.py
"""

from pathlib import Path

from pypdf import PdfReader


PDFS_DIR = Path("data/pdfs")
OUTPUT_PATH = Path("data/processed_pdfs.txt")

SUPPORTED_EXTENSIONS = [".pdf"]


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from a PDF page.

    Args:
        text: Raw text returned by pypdf.

    Returns:
        Cleaned text with empty lines removed. Returns an empty string when
        no text is available.
    """
    if not text:
        return ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from one PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        Text extracted from all readable pages, including source and page
        metadata in plain text.

    Raises:
        Exception: Propagates errors raised by pypdf when the file cannot be
        opened or parsed.
    """
    reader = PdfReader(str(path))

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        cleaned = clean_text(raw_text)

        if cleaned:
            pages_text.append(
                f"Fonte: {path.name}\nPágina: {page_number}\n\n{cleaned}"
            )

    return "\n\n" + ("=" * 80) + "\n\n".join(pages_text)


def build_documents_from_pdfs() -> None:
    """
    Process all PDF files in data/pdfs and write extracted text to disk.

    This loader is optional in the current project, but it allows the RAG
    pipeline to be extended with PDF-based IBGE publications.

    Raises:
        FileNotFoundError: If data/pdfs does not exist or contains no PDFs.
        ValueError: If no text can be extracted from the available PDFs.
    """
    if not PDFS_DIR.exists():
        raise FileNotFoundError("A pasta data/pdfs não existe.")

    files = [
        path
        for path in PDFS_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError("Nenhum PDF encontrado em data/pdfs.")

    all_texts = []

    for path in files:
        print(f"Processando PDF: {path.name}")

        try:
            text = extract_text_from_pdf(path)

            if text.strip():
                all_texts.append(text)

        except Exception as error:
            print(f"Erro ao processar {path.name}: {error}")

    final_text = "\n\n".join(all_texts)

    if not final_text.strip():
        raise ValueError("Nenhum texto foi extraído dos PDFs.")

    OUTPUT_PATH.write_text(final_text, encoding="utf-8")

    print(f"\nArquivo gerado com sucesso: {OUTPUT_PATH}")
    print(f"Total de caracteres: {len(final_text)}")


if __name__ == "__main__":
    build_documents_from_pdfs()