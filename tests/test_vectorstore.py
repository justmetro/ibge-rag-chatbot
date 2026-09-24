from pathlib import Path

from langchain_core.documents import Document

from src.pdf_loader import extract_text_from_pdf
from src.vectorstore import (
    format_source_label,
    parse_block_header,
    split_documents,
    split_text_into_blocks,
)


TABLE_BLOCK = """Fonte: Tabela 1.3 (IndicSexCor_BR).xls
Aba: 2024

Conteúdo extraído da tabela:
- Tabela 1.3 - Indicadores segundo sexo e cor ou raça - Brasil - 2024
- Homem | 3.500
- Mulher | 2.800"""


def test_parse_block_header_reads_source_and_sheet():
    header = parse_block_header(TABLE_BLOCK)

    assert header == {
        "source": "Tabela 1.3 (IndicSexCor_BR).xls",
        "sheet": "2024",
        "title": "Tabela 1.3 - Indicadores segundo sexo e cor ou raça - Brasil - 2024",
    }


def test_format_source_label_includes_sheet():
    label = format_source_label({"source": "Tabela 1.1 (Indic_BR).xls", "sheet": "2024"})

    assert label == "Tabela 1.1 (Indic_BR).xls (aba 2024)"


def test_split_documents_prefixes_title_on_continuation_chunks():
    block = TABLE_BLOCK + "\n" + "\n".join(f"- Linha {i} | {i * 10}" for i in range(200))
    document = Document(
        page_content=block,
        metadata={
            "source": "Tabela 1.3 (IndicSexCor_BR).xls",
            "sheet": "2024",
            "title": "Tabela 1.3 - Indicadores segundo sexo e cor ou raça",
        },
    )

    chunks = split_documents([document])

    assert len(chunks) > 1
    assert chunks[0].page_content.startswith("Fonte:")
    assert all(
        chunk.page_content.startswith("[Tabela 1.3 - Indicadores segundo sexo e cor ou raça]")
        for chunk in chunks[1:]
    )


def test_pdf_pages_are_split_into_separate_blocks(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("Primeira página"), FakePage("Segunda página")]

    monkeypatch.setattr("src.pdf_loader.PdfReader", FakeReader)

    blocks = split_text_into_blocks(extract_text_from_pdf(Path("relatorio.pdf")))

    assert len(blocks) == 2
    assert parse_block_header(blocks[1]) == {"source": "relatorio.pdf", "page": "2"}


def test_split_documents_merges_short_tail_chunk_into_previous():
    rows = "\n".join(f"- Linha {i} | {i * 10}" for i in range(60))
    block = TABLE_BLOCK + "\n" + rows + "\n- Nota: dados preliminares."
    document = Document(page_content=block, metadata={"source": "Tabela.xls"})

    chunks = split_documents([document])

    assert all(len(chunk.page_content) >= 300 for chunk in chunks)
    assert chunks[-1].page_content.endswith("- Nota: dados preliminares.")
