"""
Text helpers shared by query expansion, filters and evaluation.

Keyword rules must match whole words: a plain substring test would make
"cor" match "recortes" or "corresponde", triggering the wrong expansion.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Lowercase text and remove accents.

    Args:
        text: Input text.

    Returns:
        Lowercase text without diacritics ("Rondônia" -> "rondonia").
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())

    return "".join(char for char in decomposed if not unicodedata.combining(char))


def contains_any_term(text: str, terms: list[str]) -> bool:
    """
    Check whether any term appears in the text as a whole word or phrase.

    Matching ignores case and accents, so "raca" matches "raça".

    Args:
        text: Text to inspect.
        terms: Words or phrases to look for.

    Returns:
        True if at least one term is found as a whole word; False otherwise.
    """
    normalized_text = normalize_text(text)

    return any(
        re.search(rf"(?<!\w){re.escape(normalize_text(term))}(?!\w)", normalized_text)
        for term in terms
    )
