from types import SimpleNamespace

import pytest
from google.genai import errors
from langchain_core.documents import Document

from src import rag
from src.rag import RAGBot


DOCS = [
    Document(
        page_content="Homem | 3.500,00\nMulher | 2.800,00",
        metadata={"source": "Tabela 1.6 (IndicSexCor_BR_Rend).xls", "sheet": "2024"},
    )
]


class FakeModels:
    """Return a scripted outcome per model: answer text or an error to raise."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes[model]

        if isinstance(outcome, Exception):
            raise outcome

        return SimpleNamespace(text=outcome)


def make_bot(outcomes, monkeypatch):
    monkeypatch.setattr(rag, "GEMINI_MODELS", list(outcomes))
    models = FakeModels(outcomes)
    bot = RAGBot.__new__(RAGBot)
    bot.client = SimpleNamespace(models=models)
    bot._respostas = {}
    return bot, models


def quota_error():
    return errors.ClientError(
        429,
        {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
    )


def api_key_error():
    return errors.ClientError(
        400,
        {"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}},
    )


def test_responder_com_gemini_sends_context_and_system_prompt(monkeypatch):
    bot, models = make_bot({"modelo-a": "Homens: R$ 3.500,00."}, monkeypatch)

    answer = bot._responder_com_gemini("Compare homens e mulheres", DOCS)

    assert answer == "Homens: R$ 3.500,00."

    call = models.calls[0]

    assert "Compare homens e mulheres" in call["contents"]
    assert "Tabela 1.6 (IndicSexCor_BR_Rend).xls (aba 2024)" in call["contents"]
    assert "IBGE" in call["config"].system_instruction


def test_responder_com_gemini_tries_next_model_when_quota_is_exhausted(monkeypatch):
    bot, models = make_bot(
        {"modelo-a": quota_error(), "modelo-b": "Resposta do segundo modelo."},
        monkeypatch,
    )

    answer = bot._responder_com_gemini("Pergunta", DOCS)

    assert answer == "Resposta do segundo modelo."
    assert [call["model"] for call in models.calls] == ["modelo-a", "modelo-b"]


def test_responder_com_gemini_stops_on_invalid_api_key(monkeypatch):
    bot, models = make_bot(
        {"modelo-a": api_key_error(), "modelo-b": "Nunca chamado."},
        monkeypatch,
    )

    with pytest.raises(errors.ClientError):
        bot._responder_com_gemini("Pergunta", DOCS)

    assert len(models.calls) == 1


def test_ask_reuses_cached_answer_for_repeated_question(monkeypatch):
    bot, models = make_bot({"modelo-a": "Resposta."}, monkeypatch)
    monkeypatch.setattr(rag, "retrieve_documents", lambda **kwargs: DOCS)

    first = bot.ask("Compare homens e mulheres")
    second = bot.ask("compare HOMENS e mulheres ")

    assert first["answer"] == second["answer"] == "Resposta."
    assert len(models.calls) == 1
