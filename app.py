import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="HR Assistant Chatbot", page_icon="💼")

# ---------- Config ----------
CORPUS_PATH = "zyro-dynamics-hr-corpus/"  # put your HR policy PDFs in this folder in the repo

# Reads the key from Streamlit Cloud "Secrets" (Settings -> Secrets) or a local .env
try:
    LLM_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    LLM_API_KEY = os.getenv("GROQ_API_KEY")

if not LLM_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Secrets or a .env file.")
    st.stop()


# ---------- Build the pipeline once, cache it ----------
@st.cache_resource(show_spinner="Setting up the knowledge base (first load only)...")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm_model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        max_tokens=500,
        groq_api_key=LLM_API_KEY,
    )

    RAG_PROMPT = ChatPromptTemplate.from_template(
        """
        You are an HR assistant. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't have that information."

        context : {context},
        question : {question}
        """
    )

    GUARDIAL_PROMPT = ChatPromptTemplate.from_template(
        """
        You are a scope classifier for an HR assistant. Decide whether the question
        below is something an HR assistant should answer (company leave policy,
        reimbursement, code of conduct, etc.). Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.

        Question: {question}
        """
    )

    def format_docs(docs):
        return ".\n\n".join(d.page_content for d in docs)

    def rag_chain(question: str):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        chain = RAG_PROMPT | llm_model | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer, "sources": docs}

    REFUSAL_MESSAGE = (
        "I'm an HR assistant and can only help with questions about company HR "
        "policies (leave, reimbursement, code of conduct, etc.). I don't have "
        "information to answer that question."
    )

    def ask_bot(question: str):
        guardial_chain = GUARDIAL_PROMPT | llm_model | StrOutputParser()
        verdict = guardial_chain.invoke({"question": question}).strip().upper()
        if "OUT_OF_SCOPE" in verdict:
            return {"answer": REFUSAL_MESSAGE, "sources": []}
        return rag_chain(question)

    # number of unique PDF files (not pages)
    n_files = len({os.path.basename(d.metadata.get("source", "")) for d in documents})

    return ask_bot, n_files, len(chunks)


ask_bot, num_docs, num_chunks = build_pipeline()


# ---------- Helpers for source citations ----------
def format_sources(docs, answer):
    """Return a sorted list of (file_name, page_number). Empty if the bot refused."""
    normalized = answer.replace("\u2019", "'")  # handle curly apostrophe
    if "I don't have" in normalized:
        return []
    return sorted({
        (os.path.basename(d.metadata.get("source", "unknown")),
         int(d.metadata.get("page", 0)) + 1)
        for d in docs
    })


def show_sources(sources):
    if sources:
        with st.expander("📄 Sources"):
            for name, page in sources:
                st.write(f"- {name} (page {page})")


# ---------- UI ----------
st.title("💼 HR Assistant Chatbot")
st.caption(f"Answers grounded in {num_docs} company HR documents ({num_chunks} chunks indexed).")

if "history" not in st.session_state:
    st.session_state.history = []

# history items: (role, text, sources)
for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        show_sources(sources)

question = st.chat_input("Ask an HR policy question...")

if question:
    st.session_state.history.append(("user", question, []))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_bot(question)
        sources = format_sources(result["sources"], result["answer"])
        st.write(result["answer"])
        show_sources(sources)
    st.session_state.history.append(("assistant", result["answer"], sources))