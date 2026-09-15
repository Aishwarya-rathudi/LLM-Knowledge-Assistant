import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# ---------------------------------------------------------------------------
# Page config + light theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind — Ask Your Documents",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #1f2937;
        color: #93c5fd;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .answer-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }
    .chunk-card {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .timing {
        color: #6b7280;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">📄 DocMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Ask questions grounded strictly in your document — powered by RAG + Groq</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load & index the document (cached in session so it only runs once)
# ---------------------------------------------------------------------------
PDF_PATH = "data/attention.pdf"  # swap in any PDF path

if "vectors" not in st.session_state:
    with st.spinner("📚 Reading and indexing the document..."):
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFLoader(PDF_PATH)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

    st.success(f"Indexed **{os.path.basename(PDF_PATH)}** — ready to answer questions.")

# ---------------------------------------------------------------------------
# LLM + chain setup
# ---------------------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500,
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ---------------------------------------------------------------------------
# Query input + answer
# ---------------------------------------------------------------------------
user_prompt = st.text_input("💬 Ask something about the document", placeholder="e.g. What is multi-head attention?")

if user_prompt:
    with st.spinner("Thinking..."):
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        elapsed = time.process_time() - start

    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(f"**Answer**")
    st.write(response["answer"])
    st.markdown(f'<div class="timing">⏱ Answered in {elapsed:.2f}s</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(f"🔎 View {len(response['context'])} source chunks used"):
        for i, doc in enumerate(response["context"]):
            page = doc.metadata.get("page", "?")
            st.markdown(
                f'<span class="source-badge">Chunk {i + 1} · page {page}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="chunk-card">{doc.page_content}</div>', unsafe_allow_html=True)