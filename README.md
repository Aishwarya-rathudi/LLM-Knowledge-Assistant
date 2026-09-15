# LLM Knowledge Assistant

A hands-on collection of LangChain experiments exploring chatbots, LLM API serving, and Retrieval-Augmented Generation (RAG) — building toward a unified knowledge assistant that can answer questions grounded in documents, the web, Wikipedia, and arXiv.


## What's in this repo

| File | Folder (suggested) | What it does |
|---|---|---|
| `localama.py` | `chatbot/` | Streamlit chatbot powered by a **local Llama 2** model via Ollama. |
| `app.py` (Groq chatbot) | `chatbot/` | Streamlit chatbot powered by **Groq's `llama-3.3-70b-versatile`**, with LangSmith tracing enabled. |
| `app.py` (LangServe API) | `api/` | FastAPI + **LangServe** server exposing chains as REST endpoints: `/groq` (raw ChatGroq), `/essay` (100-word essay generator), and `/poem` (children's poem generator) — all powered by Groq models. |
| `client.py` | `api/` | Streamlit front-end that calls the LangServe `/essay` and `/poem` endpoints above. |
| `app.py` (RAG demo) | `rag/` | Streamlit RAG app: loads a document, embeds it with Ollama's `nomic-embed-text`, indexes it in **FAISS**, and answers questions using **Groq**. Currently demoed against `attention.pdf` ("Attention Is All You Need"). |
| `simplerag.ipynb` | `rag/` | 📝 Empty notebook — scaffold for a basic RAG walkthrough (using `attention.pdf` / `speech.txt` as source documents). |
| `agents.ipynb` | `agents/` | 📝 Empty notebook — scaffold for LangChain agent experiments. |
| `attention.pdf` | `data/` | Sample PDF ("Attention Is All You Need") used to demo document-grounded Q&A. |
| `speech.txt` | `data/` | Sample plain-text document (Wilson's WWI address to Congress) for testing text ingestion/RAG. |
| `main.py` | root | Placeholder entry point for the eventual unified assistant. |


## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangChain, LangChain Community, LangChain Core |
| LLM Providers | **Groq** (free tier, production models), Ollama (local Llama 2) |
| Embeddings | Ollama (`nomic-embed-text`), HuggingFace (`sentence-transformers`) |
| Vector Stores | FAISS, ChromaDB, Cassandra (via CassIO) |
| Retrieval Sources | PDFs, web pages, Wikipedia, arXiv |
| Document Parsing | PyPDF, PyPDF2, BeautifulSoup4 |
| API Serving | FastAPI, LangServe, SSE-Starlette, Uvicorn |
| Frontend | Streamlit |
| Tracing | LangSmith |


## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- A free [Groq API key](https://console.groq.com/keys) (no credit card required)
- [Ollama](https://ollama.com/) installed locally, with the following models pulled, for scripts that use local models/embeddings:
  ```bash
  ollama pull llama2
  ollama pull nomic-embed-text
  ```

## Installation

### Using uv (recommended)

```bash
git clone <your-repo-url>
cd llm-knowledge-assistant
uv sync
```

### Using pip

```bash
git clone <your-repo-url>
cd llm-knowledge-assistant
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY="your-groq-api-key"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="LLM-Knowledge-Assistant"
```


## Running the demos

Each script is a standalone demo — run the ones relevant to what you want to try.

### 1. Local chatbot (Ollama / Llama 2)
```bash
streamlit run chatbot/localama.py
```
Requires Ollama running locally with `llama2` pulled.

### 2. Groq chatbot
```bash
streamlit run chatbot/app.py
```
Requires `GROQ_API_KEY` in `.env`. Uses `llama-3.3-70b-versatile`.

### 3. LangServe API + client
Start the API server:
```bash
python api/app.py
```
This serves endpoints at `http://localhost:8000` (`/groq`, `/essay`, `/poem`), all powered by Groq models. In a separate terminal, launch the client UI:
```bash
streamlit run api/client.py
```

### 4. RAG demo (document Q&A)
```bash
streamlit run rag/app.py
```
Requires `GROQ_API_KEY` in `.env` and Ollama running locally (for the `nomic-embed-text` embeddings). Loads and indexes a document — currently set up to demo against pdf — and answers questions grounded strictly in that document, showing the retrieved source chunks alongside each answer.

### 5. Notebooks
Open `rag/simplerag.ipynb` or `agents/agents.ipynb` in Jupyter/VS Code to experiment interactively. Both are currently empty starting points.

## Sample Data

- `data/attention.pdf` — the paper, used to demo the RAG pipeline's ability to answer questions grounded strictly in the document (with retrieved source chunks shown for transparency).
- `data/speech.txt` — a plain-text speech, useful for testing simple text ingestion and chunking.
