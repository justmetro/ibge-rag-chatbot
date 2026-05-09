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
                /* Fundo geral */
                .stApp {
                    background: linear-gradient(180deg, #081225 0%, #0b1730 100%);
                    color: #e5e7eb;
                }

                .block-container {
                    padding-top: 3rem;
                    padding-bottom: 2rem;
                }

                /* Sidebar */
                section[data-testid="stSidebar"] {
                    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
                    border-right: 1px solid #1e293b;
                }

                section[data-testid="stSidebar"] * {
                    color: #e5e7eb !important;
                }

                h1, h2, h3, h4, p, li, span, label {
                    color: #e5e7eb !important;
                }

                /* Alert/info */
                div[data-testid="stAlert"] {
                    background: rgba(37, 99, 235, 0.18) !important;
                    border: 1px solid #3b82f6 !important;
                    border-radius: 14px !important;
                }

                div[data-testid="stAlert"] * {
                    color: #e5e7eb !important;
                }

                /* Chat messages */
                div[data-testid="stChatMessage"] {
                    background-color: rgba(15, 23, 42, 0.78);
                    border: 1px solid #22304a;
                    border-radius: 16px;
                    padding: 0.85rem;
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
                }

                div[data-testid="stChatMessage"] * {
                    color: #e5e7eb !important;
                }

                /* Expander */
                [data-testid="stExpander"] {
                    background-color: rgba(15, 23, 42, 0.82);
                    border: 1px solid #334155;
                    border-radius: 12px;
                }

                [data-testid="stExpander"] * {
                    color: #e5e7eb !important;
                }

                pre {
                    background-color: #0b1220 !important;
                    color: #d1d5db !important;
                    border: 1px solid #263247;
                    border-radius: 12px;
                }

                code {
                    color: #93c5fd !important;
                    background-color: transparent !important;
                }

                hr {
                    border-color: #253249 !important;
                }

                div[data-testid="stCaptionContainer"] {
                    color: #9ca3af !important;
                }

                header[data-testid="stHeader"],
                footer {
                    background-color: transparent !important;
                }

                [data-testid="stBottom"] {
                    background-color: #0b1730 !important;
                }

                [data-testid="stBottom"] > div {
                    background-color: #0b1730 !important;
                }

                /* Caixa externa do chat input */
                div[data-testid="stChatInput"] {
                    background: #0b1220 !important;
                    border: 1px solid #1e3a5f !important;
                    border-radius: 18px !important;
                    padding: 0.55rem !important;
                    box-shadow:
                        0 0 0 1px rgba(59, 130, 246, 0.10),
                        0 10px 28px rgba(0, 0, 0, 0.28) !important;
                }

                /* Remove fundo branco interno do Streamlit */
                div[data-testid="stChatInput"] > div,
                div[data-testid="stChatInput"] > div > div,
                div[data-testid="stChatInput"] form,
                div[data-testid="stChatInput"] form > div {
                    background: #0b1220 !important;
                }

                /* INPUT ATIVO - mais vibrante */
                div[data-testid="stChatInput"] textarea {
                    background: linear-gradient(180deg, #17345d 0%, #10284a 100%) !important;
                    color: #ffffff !important;
                    border: 1.5px solid #60a5fa !important;
                    border-radius: 999px !important;
                    padding: 0.8rem 1rem !important;
                    box-shadow:
                        0 0 0 1px rgba(59, 130, 246, 0.20),
                        0 0 18px rgba(59, 130, 246, 0.18);
                }

                div[data-testid="stChatInput"] textarea:focus {
                    border: 1.5px solid #60a5fa !important;
                    box-shadow:
                        0 0 0 2px rgba(96, 165, 250, 0.18),
                        0 0 24px rgba(59, 130, 246, 0.24) !important;
                    outline: none !important;
                }

                div[data-testid="stChatInput"] textarea::placeholder {
                    color: #cbd5e1 !important;
                    opacity: 0.95 !important;
                }

                /* INPUT BLOQUEADO */
                div[data-testid="stChatInput"] textarea:disabled,
                div[data-testid="stChatInput"] textarea[disabled] {
                    background: linear-gradient(180deg, #1a2740 0%, #162338 100%) !important;
                    color: #ffffff !important;
                    -webkit-text-fill-color: #ffffff !important;
                    border: 1.5px solid #64748b !important;
                    opacity: 1 !important;
                    cursor: not-allowed !important;
                    box-shadow: none !important;
                }

                div[data-testid="stChatInput"] textarea:disabled::placeholder,
                div[data-testid="stChatInput"] textarea[disabled]::placeholder {
                    color: #f8fafc !important;
                    -webkit-text-fill-color: #f8fafc !important;
                    opacity: 0.92 !important;
                }

                /* Botão enviar */
                div[data-testid="stChatInput"] button {
                    background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%) !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 999px !important;
                    width: 42px !important;
                    height: 42px !important;
                    box-shadow: 0 0 14px rgba(59, 130, 246, 0.25);
                }

                div[data-testid="stChatInput"] button:hover {
                    background: linear-gradient(180deg, #60a5fa 0%, #3b82f6 100%) !important;
                }

                div[data-testid="stChatInput"] button:disabled {
                    background: #1e293b !important;
                    color: #cbd5e1 !important;
                    opacity: 1 !important;
                }

                button {
                    border-radius: 10px !important;
                }

                [data-testid="stToggle"] label {
                    color: #e5e7eb !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("assets/logo.png", width=45)

    with col2:
        st.markdown("### IBGE RAG Chatbot")

    dark_mode_value = st.toggle(
        "Modo escuro",
        value=st.session_state.dark_mode,
        key="dark_mode_toggle"
    )

    if dark_mode_value != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_value
        st.rerun()

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

aplicar_modo_escuro()


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