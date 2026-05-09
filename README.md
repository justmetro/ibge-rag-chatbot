# IBGE RAG Chatbot

Chatbot experimental para responder perguntas sobre dados sociais brasileiros usando tabelas públicas do IBGE, busca semântica e Retrieval-Augmented Generation (RAG).

## Objetivo

O objetivo do projeto é demonstrar uma aplicação prática de Python, análise de dados, embeddings, banco vetorial e interface web com Streamlit.

O sistema permite que o usuário faça perguntas sobre indicadores sociais brasileiros e retorna respostas em linguagem natural com base em trechos relevantes extraídos de tabelas públicas do IBGE.

## Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Gemini API
- Pandas
- OpenPyXL
- xlrd

## Funcionalidades

- Leitura de tabelas `.xls`, `.xlsx` e `.csv`
- Conversão das tabelas para texto pesquisável
- Criação de base vetorial com ChromaDB
- Busca semântica com embeddings locais
- Expansão simples de consultas para melhorar recuperação
- Filtro por metadata para separar indicadores e coeficientes de variação
- Integração com Gemini API para síntese em linguagem natural
- Fallback demo caso a API generativa não esteja disponível
- Interface web com Streamlit
- Exibição dos trechos e fontes utilizadas
- Modo escuro
- Bloqueio do input enquanto a resposta está sendo gerada

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
- Coeficientes de variação dos indicadores de rendimento

## Exemplos de perguntas

- Quais indicadores aparecem nas tabelas?
- As tabelas têm dados de rendimento?
- As tabelas possuem dados por sexo ou cor?
- Quais recortes territoriais aparecem?
- Compare o rendimento entre homens e mulheres.
- Mostre os coeficientes de variação dos indicadores de rendimento.
- Quais são os valores de rendimento médio real habitual do trabalho principal para Brasil, Norte e Rondônia?

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
│   ├── document_filters.py
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
Retriever ChromaDB com filtro por metadata
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
- `src/pdf_loader.py`: loader opcional para documentos PDF
- `src/embedder.py`: configuração do modelo de embeddings local
- `src/vectorstore.py`: criação, carregamento e consulta da base vetorial ChromaDB
- `src/query_expansion.py`: expansão simples de consultas para melhorar recuperação
- `src/document_filters.py`: regras de intenção para lidar com coeficientes de variação
- `src/rag.py`: orquestração do fluxo RAG
- `src/prompts.py`: prompts utilizados pelo Gemini e mensagens de fallback
- `scripts/evaluate_retrieval.py`: avaliação da recuperação semântica

## Decisões técnicas

### Uso de tabelas do IBGE

O projeto foi adaptado para trabalhar com tabelas `.xls`, `.xlsx` e `.csv`, pois muitos dados públicos do IBGE são disponibilizados em formato tabular. As tabelas são convertidas para texto estruturado antes de serem indexadas no banco vetorial.

### Embeddings locais

Foi utilizado o modelo `sentence-transformers/all-MiniLM-L6-v2`, por ser leve, gratuito e adequado para protótipos de busca semântica. Isso reduz custos e permite que a etapa de recuperação funcione sem depender de API externa.

### ChromaDB

O ChromaDB foi escolhido como banco vetorial por ser simples de configurar, persistente localmente e bem integrado ao ecossistema LangChain.

### Gemini API com fallback

Quando a variável `GEMINI_API_KEY` está configurada, o app usa Gemini para sintetizar respostas em linguagem natural com base nos trechos recuperados. Caso a API não esteja disponível ou atinja limite temporário de uso, o sistema mantém um modo demo que retorna uma síntese simples e os trechos relevantes.

### Expansão simples de consulta

Foi adicionada uma etapa de expansão simples de consulta para melhorar a recuperação em perguntas sobre sexo, cor ou raça, rendimento, valores monetários, recortes territoriais e coeficientes de variação.

Essa estratégia ajudou a recuperar melhor tabelas específicas, como comparações entre homens e mulheres ou perguntas sobre rendimento por Brasil, Norte e Rondônia.

### Filtro por metadata

As tabelas de coeficientes de variação são identificadas durante a criação dos documentos e recebem metadata específica no ChromaDB.

Assim, quando o usuário pede valores de rendimento, o sistema evita recuperar coeficientes de variação como se fossem valores monetários. Quando o usuário pede explicitamente coeficientes, essas tabelas são incluídas na busca.

## Avaliação do retriever

Foi criado um script de avaliação em `scripts/evaluate_retrieval.py`, com perguntas representativas sobre os dados carregados.

A avaliação verifica:

- Quantidade de documentos retornados
- Presença de termos esperados nos trechos recuperados
- Precision@k
- Recall proxy
- MRR
- NDCG
- Latência média

Resultado atual da avaliação:

```text
Total de testes: 6
Testes aprovados: 6
Taxa de sucesso simples: 100%
Precision@k médio: 1.000
Recall proxy médio: 1.000
MRR médio: 1.000
NDCG médio: 1.000
Latência média: 3.134s
```

Essa avaliação usa métricas heurísticas baseadas em termos esperados. Ela não substitui um benchmark formal rotulado, mas ajuda a monitorar qualidade de recuperação e latência durante o desenvolvimento.

## Testes e CI

O projeto possui testes automatizados com `pytest`, cobrindo partes centrais da lógica do RAG:

- Expansão de consultas
- Filtros de documentos recuperados
- Métricas da avaliação do retriever

Atualmente, a suíte possui 16 testes unitários.

Também foi configurado um workflow de GitHub Actions em `.github/workflows/tests.yml`, que executa os testes automaticamente a cada `push` ou `pull request`.

Para rodar os testes localmente:

```bash
pytest
```

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

Rode a avaliação do retriever:

```bash
python scripts/evaluate_retrieval.py
```

Rode o app:

```bash
streamlit run app.py
```

## Variáveis de ambiente

Crie um arquivo `.env` local com:

```env
GEMINI_API_KEY=sua_chave_aqui
RETRIEVER_K=5
```

A chave Gemini é opcional. Sem ela, o app funciona em modo demo.

## Status do projeto

Versão atual: MVP funcional com deploy público.

Nesta versão, o sistema recupera trechos relevantes das tabelas carregadas, utiliza Gemini API para sintetizar respostas em linguagem natural quando a variável `GEMINI_API_KEY` está configurada, e mantém um modo demo como fallback caso a API não esteja disponível.

## Cuidados de segurança

Este projeto não versiona arquivos `.env` nem chaves de API.

## Limitações e próximos passos

A versão atual já possui deploy público, recuperação semântica com ChromaDB, integração com Gemini API e fallback demo caso a chave da API não esteja disponível.

Apesar disso, o projeto ainda pode evoluir em alguns pontos:

- Criar testes unitários com `pytest`
- Adicionar CI/CD com GitHub Actions
- Testar diferentes tamanhos de chunk e valores de `k` no retriever
- Melhorar o tratamento de tabelas complexas do IBGE, especialmente cabeçalhos com múltiplas linhas
- Incluir mais bases públicas sobre educação, trabalho, rendimento e desigualdade
- Exibir gráficos e tabelas resumidas diretamente na interface
- Explorar deploy alternativo em Hugging Face Spaces

## Aprendizados e insights

### Tabelas reais exigem tratamento diferente de PDFs

O projeto começou como um RAG voltado para documentos, mas as fontes públicas do IBGE frequentemente aparecem em formato tabular. Por isso, foi necessário transformar planilhas `.xls`, `.xlsx` e `.csv` em texto estruturado antes da indexação.

Essa decisão tornou o projeto mais alinhado com aplicações reais de dados públicos.

### Query expansion melhorou a recuperação

Perguntas curtas como “As tabelas possuem dados por sexo ou cor?” nem sempre recuperavam os melhores trechos inicialmente.

A solução foi adicionar expansão simples de consulta com termos como `sexo`, `homem`, `mulher`, `cor ou raça`, `rendimento` e `trabalho principal`.

Após essa melhoria, a avaliação passou a recuperar corretamente os termos esperados para as perguntas de teste.

### Metadata reduziu ruído na recuperação

As tabelas de coeficientes de variação apareciam em perguntas sobre valores de rendimento, confundindo o contexto enviado ao Gemini.

Para resolver isso, os documentos passaram a receber metadata indicando se pertencem ou não a tabelas de coeficientes. O retriever usa essa informação para filtrar resultados quando a pergunta pede valores monetários, mas preserva os coeficientes quando eles são solicitados explicitamente.

### Embeddings locais são suficientes para o MVP

O uso de embeddings locais com `sentence-transformers/all-MiniLM-L6-v2` foi suficiente para o tamanho atual da base. Isso permitiu manter a recuperação semântica sem custo de API, deixando a API generativa apenas para a síntese final da resposta.

### Avaliação simples já ajuda a encontrar falhas reais

Mesmo uma avaliação heurística baseada em termos esperados foi útil para identificar falhas na recuperação, como perguntas sobre sexo e cor ou mistura entre valores de rendimento e coeficientes de variação.

A avaliação evoluiu para incluir Precision@k, Recall proxy, MRR e NDCG, tornando o acompanhamento da qualidade do retriever mais transparente.