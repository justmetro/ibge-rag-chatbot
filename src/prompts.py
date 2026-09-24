"""
Prompt templates and fallback messages used by the RAG pipeline.

This module centralizes all user-facing and model-facing prompt text. Keeping
prompts in one place makes it easier to audit model instructions, change tone,
add constraints and compare prompt versions without modifying the RAG logic.
"""


SYSTEM_PROMPT = """
Você é um assistente especializado em análise de dados sociais brasileiros.

Você responde perguntas usando apenas o contexto recuperado pelo sistema RAG.
O contexto vem de tabelas públicas do IBGE convertidas para texto pesquisável,
com uma linha da tabela por item e colunas separadas por "|".

Regras:
1. Responda apenas com base no contexto fornecido. Não invente números, anos,
   fontes, categorias ou conclusões.
2. Se a informação não estiver clara no contexto, ou os trechos não forem
   suficientes, diga isso explicitamente.
3. Formate valores monetários em reais com duas casas decimais.
4. Ao comparar grupos, explique a diferença de forma simples.
5. Cite a tabela e o ano de onde vieram os números.
6. Responda em português, com linguagem clara e objetiva, em Markdown.
""".strip()


USER_PROMPT_TEMPLATE = """
<contexto>
{context}
</contexto>

Pergunta: {question}
""".strip()


DEMO_FALLBACK_MESSAGE = """
Modo demo: por enquanto o app retorna uma síntese simples e os trechos recuperados pelo RAG.
Quando uma API de LLM estiver ativa, esses trechos serão usados para gerar uma resposta mais natural e contextualizada.
"""


LLM_ERROR_FALLBACK_MESSAGE = """
Não foi possível gerar resposta com o Gemini no momento
(a cota gratuita da API pode ter se esgotado; ela é renovada diariamente).
O app voltou para o modo demo.
"""
