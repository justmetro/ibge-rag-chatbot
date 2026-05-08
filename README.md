# IBGE RAG Chatbot

Chatbot experimental para responder perguntas sobre dados sociais brasileiros usando tabelas públicas do IBGE, busca semântica e Retrieval-Augmented Generation (RAG).

## Objetivo

O objetivo do projeto é demonstrar uma aplicação prática de Python, análise de dados, embeddings, banco vetorial e interface web com Streamlit.

O sistema permite que o usuário faça perguntas sobre indicadores sociais brasileiros e retorna trechos relevantes extraídos de tabelas públicas do IBGE.

## Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Pandas
- OpenPyXL
- xlrd

## Funcionalidades

- Leitura de tabelas `.xls`, `.xlsx` e `.csv`
- Conversão das tabelas para texto pesquisável
- Criação de base vetorial com ChromaDB
- Busca semântica com embeddings locais
- Interface web com Streamlit
- Exibição dos trechos e fontes utilizadas
- Modo demo sem dependência de API paga

## Deploy

A aplicação está disponível em:

https://ibge-rag-chatbot-justmetro.streamlit.app/

## Screenshots

### Tela inicial

![Tela inicial do app](assets/home.png)

### Exemplo de resposta

![Exemplo de resposta do chatbot](assets/resposta.png)

## Dados utilizados

O projeto utiliza tabelas públicas do IBGE relacionadas a indicadores sociais brasileiros, incluindo informações sobre rendimento, trabalho, sexo, cor ou raça e recortes territoriais.

Exemplos de tabelas carregadas:

- Indicadores por Brasil
- Indicadores por Grandes Regiões e Unidades da Federação
- Indicadores por sexo e cor ou raça
- Indicadores de rendimento do trabalho

## Exemplos de perguntas

- Quais indicadores aparecem nas tabelas?
- As tabelas têm dados de rendimento?
- As tabelas possuem dados por sexo ou cor?
- Quais recortes territoriais aparecem?

## Estrutura do projeto

```text
ibge-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   ├── home.png
│   ├── resposta.png
│   └── logo.png
├── data/
│   ├── documents.txt
│   ├── pdfs/
│   └── tables/
├── reports/
│   └── retrieval_evaluation.txt
├── scripts/
│   └── evaluate_retrieval.py
├── src/
│   ├── rag.py
│   ├── vectorstore.py
│   ├── table_loader.py
│   ├── pdf_loader.py
│   ├── prompts.py
│   ├── embedder.py
│   └── query_expansion.py
└── .streamlit/
    └── config.toml
```

## Arquitetura técnica

O projeto segue uma arquitetura RAG simples e modular:

```text
Usuário
  ↓
Interface Streamlit
  ↓
RAGBot
  ↓
Expansão simples de consulta
  ↓
Retriever ChromaDB
  ↓
Trechos relevantes das tabelas
  ↓
Gemini API
  ↓
Resposta em linguagem natural + fontes utilizadas
```

Principais módulos:

- `app.py`: interface web com Streamlit
- `src/table_loader.py`: leitura e conversão de tabelas do IBGE em texto pesquisável
- `src/embedder.py`: configuração do modelo de embeddings local
- `src/vectorstore.py`: criação e carregamento da base vetorial ChromaDB
- `src/query_expansion.py`: expansão simples de consultas para melhorar recuperação
- `src/rag.py`: orquestração do fluxo RAG
- `src/prompts.py`: prompts utilizados pelo Gemini e mensagens de fallback
- `scripts/evaluate_retrieval.py`: avaliação simples da recuperação semântica

## Decisões técnicas

### Uso de tabelas do IBGE

O projeto foi adaptado para trabalhar com tabelas `.xls`, `.xlsx` e `.csv`, pois muitos dados públicos do IBGE são disponibilizados em formato tabular. As tabelas são convertidas para texto estruturado antes de serem indexadas no banco vetorial.

### Embeddings locais

Foi utilizado o modelo `sentence-transformers/all-MiniLM-L6-v2`, por ser leve, gratuito e adequado para protótipos de busca semântica. Isso reduz custos e permite que a etapa de recuperação funcione sem depender de API externa.

### ChromaDB

O ChromaDB foi escolhido como banco vetorial por ser simples de configurar, persistente localmente e bem integrado ao ecossistema LangChain.

### Gemini API com fallback

Quando a variável `GEMINI_API_KEY` está configurada, o app usa Gemini para sintetizar respostas em linguagem natural com base nos trechos recuperados. Caso a API não esteja disponível, o sistema mantém um modo demo que retorna síntese simples e trechos relevantes.

### Expansão simples de consulta

Foi adicionada uma etapa de expansão simples de consulta para melhorar a recuperação em perguntas sobre sexo, cor ou raça, rendimento e recortes territoriais. Essa estratégia ajudou a recuperar melhor tabelas específicas, como comparações entre homens e mulheres.

## Avaliação do retriever

Foi criado um script simples de avaliação em `scripts/evaluate_retrieval.py`, com perguntas representativas sobre os dados carregados.

A avaliação verifica:

- Quantidade de documentos retornados
- Presença de termos esperados nos trechos recuperados
- Latência da recuperação
- Taxa simples de sucesso

Resultado atual da avaliação:

```text
Total de testes: 5
Testes aprovados: 5
Taxa de sucesso simples: 100%
Latência média: 0.020s
```

Essa avaliação não substitui métricas formais de Information Retrieval, mas ajuda a validar rapidamente se o retriever está retornando trechos úteis para perguntas comuns do projeto.

## Como rodar localmente

Crie e ative o ambiente virtual:

```bash
python -m venv venv
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Processe as tabelas:

```bash
python src/table_loader.py
```

Crie o índice vetorial:

```bash
python -c "from src.vectorstore import create_vectorstore; create_vectorstore(); print('Índice criado com sucesso')"
```

Rode o app:

```bash
streamlit run app.py
```

## Status do projeto

Versão atual: MVP funcional com deploy público.

Nesta versão, o sistema recupera trechos relevantes das tabelas carregadas, utiliza Gemini API para sintetizar respostas em linguagem natural quando a variável `GEMINI_API_KEY` está configurada, e mantém um modo demo como fallback caso a API não esteja disponível.

## Cuidados de segurança

Este projeto não versiona arquivos `.env` nem chaves de API.

## Limitações e próximos passos

A versão atual já possui deploy público, recuperação semântica com ChromaDB, integração com Gemini API e fallback demo caso a chave da API não esteja disponível.

Apesar disso, o projeto ainda pode evoluir em alguns pontos:

- Melhorar a avaliação do RAG com métricas mais formais de recuperação, como precision@k e recall@k
- Testar diferentes tamanhos de chunk e valores de `k` no retriever
- Melhorar o tratamento de tabelas complexas do IBGE, especialmente cabeçalhos com múltiplas linhas
- Incluir mais bases públicas sobre educação, trabalho, rendimento e desigualdade
- Exibir gráficos e tabelas resumidas diretamente na interface
- Explorar deploy alternativo em Hugging Face Spaces

## Aprendizados

Este projeto foi desenvolvido como uma aplicação prática de RAG aplicada a dados públicos brasileiros. Durante o desenvolvimento, foram trabalhados conceitos de processamento de dados, embeddings, busca semântica, banco vetorial, organização de projeto Python e construção de interface com Streamlit.