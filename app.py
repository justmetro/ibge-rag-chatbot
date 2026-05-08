import streamlit as st

from src.rag import RAGBot


st.set_page_config(
    page_title="IBGE RAG Chatbot",
    page_icon="assets/logo.png",
    layout="centered"
)


# -----------------------------
# Estado da aplicação
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# -----------------------------
# Tema escuro opcional
# -----------------------------

def aplicar_modo_escuro():
    if st.session_state.dark_mode:
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #0e1117;
                    color: #f5f5f5;
                }

                section[data-testid="stSidebar"] {
                    background-color: #161b22;
                    color: #f5f5f5;
                }

                div[data-testid="stMarkdownContainer"] {
                    color: #f5f5f5;
                }

                div[data-testid="stChatMessage"] {
                    background-color: #161b22;
                    border-radius: 12px;
                }

                div[data-testid="stTextInput"] input {
                    background-color: #262730;
                    color: #f5f5f5;
                }

                textarea {
                    background-color: #262730 !important;
                    color: #f5f5f5 !important;
                }

                code {
                    color: #9cdcfe !important;
                }

                pre {
                    background-color: #1e1e1e !important;
                    color: #f5f5f5 !important;
                }

                [data-testid="stExpander"] {
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 10px;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


aplicar_modo_escuro()


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("assets/logo.png", width=45)

    with col2:
        st.markdown("### IBGE RAG Chatbot")

    st.toggle("Modo escuro", key="dark_mode")

    st.markdown(
        """
        **Sobre o projeto**

        Este app usa RAG para buscar informações em tabelas públicas do IBGE
        sobre indicadores sociais brasileiros.

        **Modo atual:** RAG + Gemini API com fallback demo.

        **Tecnologias:**
        - Python
        - Streamlit
        - ChromaDB
        - Sentence Transformers
        - LangChain
        - Gemini API
        """
    )

    st.divider()

    st.markdown(
        """
        **Perguntas sugeridas**
        - Quais indicadores aparecem nas tabelas?
        - As tabelas têm dados de rendimento?
        - As tabelas possuem dados por sexo ou cor?
        - Quais recortes territoriais aparecem?
        - Compare o rendimento entre homens e mulheres.
        """
    )

    st.divider()

    if st.button("Limpar conversa", key="limpar_conversa_sidebar"):
        st.session_state.messages = []
        st.session_state.bot = None
        st.session_state.pending_question = None
        st.session_state.is_generating = False
        st.rerun()


# -----------------------------
# Cabeçalho
# -----------------------------

col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.image("assets/logo.png", width=70)

with col_title:
    st.title("IBGE RAG Chatbot")


st.write(
    """
    Chatbot experimental para responder perguntas sobre dados sociais brasileiros
    usando tabelas públicas do IBGE, RAG e IA generativa.
    """
)

st.info(
    "O app usa busca semântica para recuperar trechos relevantes das tabelas. "
    "Quando uma chave Gemini está configurada, a resposta é sintetizada por IA generativa; "
    "caso contrário, o app usa o modo demo."
)

st.divider()


# -----------------------------
# Utilidades
# -----------------------------

def carregar_bot():
    if st.session_state.bot is None:
        st.session_state.bot = RAGBot()
    return st.session_state.bot


def renderizar_mensagem(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("retrieved_context"):
            with st.expander("Ver trechos utilizados pelo RAG"):
                st.markdown(message["retrieved_context"])

        if message["role"] == "assistant" and message.get("sources"):
            st.caption(
                "Fontes usadas: " + ", ".join(message["sources"])
            )


# -----------------------------
# Histórico do chat
# -----------------------------

for message in st.session_state.messages:
    renderizar_mensagem(message)


# -----------------------------
# Input do usuário
# -----------------------------

placeholder = (
    "Aguarde a resposta antes de enviar outra pergunta..."
    if st.session_state.is_generating
    else "Pergunte algo sobre dados sociais brasileiros..."
)

pergunta = st.chat_input(
    placeholder,
    disabled=st.session_state.is_generating
)


if pergunta and not st.session_state.is_generating:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": pergunta,
        }
    )

    st.session_state.pending_question = pergunta
    st.session_state.is_generating = True

    st.rerun()


# -----------------------------
# Geração da resposta
# -----------------------------

if st.session_state.is_generating and st.session_state.pending_question:
    pergunta_pendente = st.session_state.pending_question

    with st.chat_message("assistant"):
        with st.spinner("Buscando trechos relevantes nas tabelas..."):
            try:
                bot = carregar_bot()
                resultado = bot.ask(pergunta_pendente)

                resposta = resultado["answer"]

                st.markdown(resposta)

                if resultado.get("retrieved_context"):
                    with st.expander("Ver trechos utilizados pelo RAG"):
                        st.markdown(resultado["retrieved_context"])

                if resultado["sources"]:
                    st.caption(
                        "Fontes usadas: " + ", ".join(resultado["sources"])
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": resposta,
                        "retrieved_context": resultado.get("retrieved_context", ""),
                        "sources": resultado.get("sources", []),
                    }
                )

            except Exception as e:
                erro = f"Erro: {e}"
                st.error(erro)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": erro,
                        "retrieved_context": "",
                        "sources": [],
                    }
                )

            finally:
                st.session_state.pending_question = None
                st.session_state.is_generating = False
                st.rerun()