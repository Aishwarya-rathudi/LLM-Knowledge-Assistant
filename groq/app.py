import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )
    st.stop()


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AskGroq — Ask Your Documents",
    page_icon="📄",
    layout="centered",
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

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
        margin-bottom: 5px;
    }

    .chunk-card {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
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


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">📄 AskGroq</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        Ask questions about your PDF using RAG + Groq + Ollama + FAISS
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. PDF PATH
# ============================================================

# app.py is inside:
#
# LLM-Knowledge-Assistant/groq/app.py
#
# PDF is inside:
#
# LLM-Knowledge-Assistant/rag/attention.pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "rag" / "attention.pdf"


# Check whether PDF actually exists

if not PDF_PATH.exists():
    st.error(f"PDF not found at:\n{PDF_PATH}")
    st.stop()


# ============================================================
# 6. CREATE EMBEDDINGS + VECTOR DATABASE
# ============================================================

if "vectors" not in st.session_state:

    with st.spinner("📚 Reading and indexing attention.pdf..."):

        try:

            # ------------------------------------------------
            # Ollama embedding model
            # ------------------------------------------------

            st.session_state.embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )


            # ------------------------------------------------
            # Load PDF
            # ------------------------------------------------

            loader = PyPDFLoader(str(PDF_PATH))

            docs = loader.load()


            # ------------------------------------------------
            # Split PDF into chunks
            # ------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            final_documents = text_splitter.split_documents(
                docs
            )


            # ------------------------------------------------
            # Create FAISS vector database
            # ------------------------------------------------

            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings,
            )


            # Store some information for display

            st.session_state.total_pages = len(docs)

            st.session_state.total_chunks = len(
                final_documents
            )


        except Exception as e:

            st.error(
                "Unable to create the vector database."
            )

            st.error(str(e))

            st.info(
                "Make sure Ollama is running and "
                "'nomic-embed-text' is installed."
            )

            st.stop()


    st.success(
        f"✅ Indexed {PDF_PATH.name} successfully — "
        f"{st.session_state.total_pages} pages / "
        f"{st.session_state.total_chunks} chunks."
    )


# ============================================================
# 7. GROQ LLM
# ============================================================

# llama-3.3-70b-versatile is not available for your account,
# so we use GPT-OSS 20B instead.

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500,
)


# ============================================================
# 8. RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:

"I could not find that information in the document."

Do not invent information.

Keep the answer clear and concise.

<context>
{context}
</context>

Question:
{input}

Answer:
"""
)


# ============================================================
# 9. CREATE DOCUMENT CHAIN
# ============================================================

document_chain = create_stuff_documents_chain(
    llm,
    prompt,
)


# ============================================================
# 10. CREATE RETRIEVER
# ============================================================

retriever = st.session_state.vectors.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 11. CREATE RETRIEVAL CHAIN
# ============================================================

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain,
)


# ============================================================
# 12. USER QUESTION
# ============================================================

user_prompt = st.text_input(
    "💬 Ask something about the document",
    placeholder="e.g. What is multi-head attention?",
)


# ============================================================
# 13. GENERATE ANSWER
# ============================================================

if user_prompt:

    with st.spinner("🔍 Searching the document..."):

        try:

            start_time = time.perf_counter()

            response = retrieval_chain.invoke(
                {
                    "input": user_prompt
                }
            )

            elapsed = time.perf_counter() - start_time


        except Exception as e:

            st.error(
                "An error occurred while generating the answer."
            )

            st.error(str(e))

            st.stop()


    # ========================================================
    # 14. DISPLAY ANSWER
    # ========================================================

    st.markdown("### 🤖 Answer")

    st.write(
        response["answer"]
    )

    st.markdown(
        f"""
        <div class="timing">
        ⏱ Answered in {elapsed:.2f} seconds
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # 15. DISPLAY SOURCE DOCUMENT CHUNKS
    # ========================================================

    context_documents = response.get(
        "context",
        []
    )

    if context_documents:

        with st.expander(
            f"🔎 View {len(context_documents)} source chunks"
        ):

            for i, doc in enumerate(
                context_documents
            ):

                # PyPDFLoader pages start from 0,
                # so add 1 for human-readable page numbers.

                page = doc.metadata.get(
                    "page",
                    "?"
                )

                if isinstance(page, int):
                    page = page + 1


                st.markdown(
                    f"""
                    <span class="source-badge">
                    Chunk {i + 1} · Page {page}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )


                # Using st.write instead of inserting
                # document text directly into HTML is safer.

                st.write(
                    doc.page_content
                )

                st.divider()


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    st.write(
        f"**File:** {PDF_PATH.name}"
    )

    if "total_pages" in st.session_state:

        st.write(
            f"**Pages:** "
            f"{st.session_state.total_pages}"
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.total_chunks}"
        )


    st.divider()


    st.header("🧠 Models")

    st.write(
        "**LLM:** Groq — GPT-OSS 20B"
    )

    st.write(
        "**Embeddings:** nomic-embed-text"
    )

    st.write(
        "**Vector DB:** FAISS"
    )