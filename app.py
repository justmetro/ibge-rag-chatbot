import streamlit as st

from src.rag import RAGBot


st.set_page_config(
    page_title="IBGE RAG Chatbot",
    page_icon="assets/logo.png",
    layout="centered"
)


with st.sidebar:
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("assets/logo.png", width=45)

    with col2:
        st.markdown("### IBGE RAG Chatbot")

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
        st.rerun()


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


if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None


def carregar_bot():
    if st.session_state.bot is None:
        st.session_state.bot = RAGBot()
    return st.session_state.bot


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


pergunta = st.chat_input("Pergunte algo sobre dados sociais brasileiros...")

if pergunta:
    st.session_state.messages.append(
        {"role": "user", "content": pergunta}
    )

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando trechos relevantes nas tabelas..."):
            try:
                bot = carregar_bot()
                resultado = bot.ask(pergunta)

                resposta = resultado["answer"]

                st.markdown(resposta)

                if resultado["sources"]:
                    st.caption(
                        "Fontes usadas: " + ", ".join(resultado["sources"])
                    )

                st.session_state.messages.append(
                    {"role": "assistant", "content": resposta}
                )

            except Exception as e:
                erro = f"Erro: {e}"
                st.error(erro)
                st.session_state.messages.append(
                    {"role": "assistant", "content": erro}
                )