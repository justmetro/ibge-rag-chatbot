"""
Table loading and text conversion utilities.

This module reads IBGE tables in CSV, XLS and XLSX formats and converts them
into a plain-text representation suitable for semantic search.

The output is written to data/documents.txt, which is later indexed by ChromaDB.
This approach allows the RAG system to retrieve relevant table rows even though
the original data comes from spreadsheets rather than natural-language documents.

Example:
    python src/table_loader.py
"""

from pathlib import Path

import pandas as pd


TABLES_DIR = Path("data/tables")
OUTPUT_PATH = Path("data/documents.txt")

SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def clean_value(value) -> str:
    """
    Convert a table cell value into a clean string.

    Args:
        value: Any value extracted from a pandas DataFrame cell.

    Returns:
        A stripped string without line breaks. Empty string is returned for
        missing values.
    """
    if pd.isna(value):
        return ""

    return str(value).replace("\n", " ").replace("\r", " ").strip()


def row_to_text(row) -> str:
    """
    Convert a DataFrame row into a pipe-separated text line.

    Args:
        row: pandas Series representing one table row.

    Returns:
        A text representation of the row. Empty string is returned when the row
        has no useful values.
    """
    values = [clean_value(value) for value in row.tolist()]
    values = [value for value in values if value and value.lower() != "nan"]

    if not values:
        return ""

    return " | ".join(values)


def summarize_raw_dataframe(
    df: pd.DataFrame,
    file_name: str,
    sheet_name: str | None = None,
    max_rows: int = 80,
) -> str:
    """
    Convert a raw DataFrame into a searchable text block.

    The function does not assume that the spreadsheet has clean headers.
    This is intentional because many IBGE spreadsheets contain title rows,
    notes, merged cells or multi-line headers. Instead of forcing a tabular
    schema, the loader keeps visible rows as text.

    Args:
        df: Raw DataFrame read from a spreadsheet or CSV file.
        file_name: Original file name used as source metadata.
        sheet_name: Optional Excel sheet name.
        max_rows: Maximum number of non-empty rows included in the output.

    Returns:
        A formatted text block with source information and extracted rows.
        Empty string is returned when no useful content is found.
    """
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return ""

    text_lines = []

    for _, row in df.iterrows():
        line = row_to_text(row)

        if line:
            text_lines.append(line)

    if not text_lines:
        return ""

    title = f"Fonte: {file_name}"

    if sheet_name:
        title += f"\nAba: {sheet_name}"

    text = []
    text.append("=" * 80)
    text.append(title)
    text.append("")
    text.append("Conteúdo extraído da tabela:")

    for line in text_lines[:max_rows]:
        text.append(f"- {line}")

    text.append("")

    return "\n".join(text)


def read_csv(path: Path) -> pd.DataFrame:
    """
    Read a CSV file using common Brazilian/Windows encodings.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the CSV content.

    Raises:
        ValueError: If the CSV cannot be read with the supported encodings.
    """
    encodings = ["utf-8", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                header=None,
            )
        except Exception:
            continue

    raise ValueError(f"Não consegui ler o CSV: {path.name}")


def process_file(path: Path) -> str:
    """
    Process one supported table file and return extracted text.

    Args:
        path: Path to a CSV, XLS or XLSX file.

    Returns:
        Text extracted from the file. Excel files may return text from multiple
        sheets joined together.
    """
    extension = path.suffix.lower()
    texts = []

    if extension == ".csv":
        df = read_csv(path)
        summary = summarize_raw_dataframe(df, path.name)

        if summary:
            texts.append(summary)

    elif extension in [".xlsx", ".xls"]:
        sheets = pd.read_excel(path, sheet_name=None, header=None)

        for sheet_name, df in sheets.items():
            summary = summarize_raw_dataframe(df, path.name, sheet_name)

            if summary:
                texts.append(summary)

    return "\n\n".join(texts)


def build_documents_from_tables() -> None:
    """
    Process all supported table files and write data/documents.txt.

    Raises:
        FileNotFoundError: If data/tables does not exist or no supported table
            files are found.
        ValueError: If no text could be extracted from the available tables.
    """
    if not TABLES_DIR.exists():
        raise FileNotFoundError("A pasta data/tables não existe.")

    files = [
        path
        for path in TABLES_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(
            "Nenhuma tabela .csv, .xlsx ou .xls encontrada em data/tables."
        )

    all_texts = []

    for path in files:
        print(f"Processando: {path.name}")

        try:
            text = process_file(path)

            if text.strip():
                all_texts.append(text)

        except Exception as error:
            print(f"Erro ao processar {path.name}: {error}")

    final_text = "\n\n".join(all_texts)

    if not final_text.strip():
        raise ValueError("Nenhum texto foi gerado a partir das tabelas.")

    OUTPUT_PATH.write_text(final_text, encoding="utf-8")

    print(f"\nArquivo gerado com sucesso: {OUTPUT_PATH}")
    print(f"Total de caracteres: {len(final_text)}")


if __name__ == "__main__":
    build_documents_from_tables()