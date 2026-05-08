from pathlib import Path
import pandas as pd


TABLES_DIR = Path("data/tables")
OUTPUT_PATH = Path("data/documents.txt")

SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def row_to_text(row):
    values = [clean_value(v) for v in row.tolist()]
    values = [v for v in values if v and v.lower() != "nan"]

    if not values:
        return ""

    return " | ".join(values)


def summarize_raw_dataframe(df, file_name, sheet_name=None, max_rows=80):
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return ""

    linhas_texto = []

    for _, row in df.iterrows():
        line = row_to_text(row)
        if line:
            linhas_texto.append(line)

    if not linhas_texto:
        return ""

    titulo = f"Fonte: {file_name}"
    if sheet_name:
        titulo += f"\nAba: {sheet_name}"

    texto = []
    texto.append("=" * 80)
    texto.append(titulo)
    texto.append("")
    texto.append("Conteúdo extraído da tabela:")

    for line in linhas_texto[:max_rows]:
        texto.append(f"- {line}")

    texto.append("")
    return "\n".join(texto)


def read_csv(path):
    encodings = ["utf-8", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                header=None
            )
        except Exception:
            continue

    raise ValueError(f"Não consegui ler o CSV: {path.name}")


def process_file(path):
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


def build_documents_from_tables():
    if not TABLES_DIR.exists():
        raise FileNotFoundError("A pasta data/tables não existe.")

    files = [
        path for path in TABLES_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError("Nenhuma tabela .csv, .xlsx ou .xls encontrada em data/tables.")

    all_texts = []

    for path in files:
        print(f"Processando: {path.name}")
        try:
            text = process_file(path)
            if text.strip():
                all_texts.append(text)
        except Exception as e:
            print(f"Erro ao processar {path.name}: {e}")

    final_text = "\n\n".join(all_texts)

    if not final_text.strip():
        raise ValueError("Nenhum texto foi gerado a partir das tabelas.")

    OUTPUT_PATH.write_text(final_text, encoding="utf-8")

    print(f"\nArquivo gerado com sucesso: {OUTPUT_PATH}")
    print(f"Total de caracteres: {len(final_text)}")


if __name__ == "__main__":
    build_documents_from_tables()