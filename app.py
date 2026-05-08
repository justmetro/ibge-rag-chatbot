import streamlit as st

from src.rag import RAGBot


st.set_page_config(
    page_title="IBGE RAG Chatbot",
    page_icon="📊",
    layout="centered"
)


with st.sidebar:
    st.title("📊 IBGE RAG Chatbot")

    st.markdown(
        """
        **Sobre o projeto**

        Este app usa RAG para buscar informações em tabelas públicas do IBGE
        sobre indicadores sociais brasileiros.

        **Modo atual:** Demo local sem API paga.

        **Tecnologias:**
        - Python
        - Streamlit
        - ChromaDB
        - Sentence Transformers
        - LangChain
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
        """
    )

    st.divider()

    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.rerun()


st.title("📊 IBGE RAG Chatbot")

st.write(
    """
    Chatbot experimental para responder perguntas sobre dados sociais brasileiros
    usando tabelas públicas do IBGE, RAG e IA generativa.
    """
)

st.info(
    "Modo demo: o app usa busca semântica para recuperar trechos relevantes das tabelas. "
    "A resposta com IA generativa poderá ser ativada depois com uma API LLM."
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