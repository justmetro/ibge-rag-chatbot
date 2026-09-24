"""
Central configuration for the RAG pipeline.

Environment variables are loaded here, once, before any other module reads
them. All data paths are resolved from the project root, so scripts work no
matter which directory they are executed from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = DATA_DIR / "tables"
PDFS_DIR = DATA_DIR / "pdfs"
DOCUMENTS_PATH = DATA_DIR / "documents.txt"
PROCESSED_PDFS_PATH = DATA_DIR / "processed_pdfs.txt"
CHROMA_PATH = PROJECT_ROOT / ".chroma"
REPORTS_DIR = PROJECT_ROOT / "reports"

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# Comma-separated list of Gemini models, tried in order. All of them are free
# on the Gemini API free tier and each one has its own quota, so when a model
# runs out (HTTP 429) the next one answers. Model IDs are retired frequently,
# so they are configurable without code changes.
GEMINI_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.8-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
    ).split(",")
    if model.strip()
]
