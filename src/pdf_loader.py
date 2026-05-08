from pathlib import Path

from pypdf import PdfReader


PDFS_DIR = Path("data/pdfs")
OUTPUT_PATH = Path("data/processed_pdfs.txt")

SUPPORTED_EXTENSIONS = [".pdf"]


def clean_text(text: str) -> str:
    """
    Limpa o texto extraído de páginas de PDF.
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
    Extrai texto de um arquivo PDF usando pypdf.
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
    Processa todos os PDFs dentro de data/pdfs e gera um arquivo textual.

    Este loader é opcional no projeto atual, mas permite evoluir o RAG
    para usar relatórios e publicações em PDF do IBGE.
    """
    if not PDFS_DIR.exists():
        raise FileNotFoundError("A pasta data/pdfs não existe.")

    files = [
        path for path in PDFS_DIR.iterdir()
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

        except Exception as e:
            print(f"Erro ao processar {path.name}: {e}")

    final_text = "\n\n".join(all_texts)

    if not final_text.strip():
        raise ValueError("Nenhum texto foi extraído dos PDFs.")

    OUTPUT_PATH.write_text(final_text, encoding="utf-8")

    print(f"\nArquivo gerado com sucesso: {OUTPUT_PATH}")
    print(f"Total de caracteres: {len(final_text)}")


if __name__ == "__main__":
    build_documents_from_pdfs()