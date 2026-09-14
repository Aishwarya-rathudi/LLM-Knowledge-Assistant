# LLM Knowledge Assistant

A hands-on collection of LangChain experiments exploring chatbots, LLM API serving, and Retrieval-Augmented Generation (RAG) — building toward a unified knowledge assistant that can answer questions grounded in documents, the web, Wikipedia, and arXiv.


## What's in this repo

| File | Folder (suggested) | What it does |
|---|---|---|
| `localama.py` | `chatbot/` | Streamlit chatbot powered by a **local Llama 2** model via Ollama. |
| `app.py` (OpenAI chatbot) | `chatbot/` | Streamlit chatbot powered by **OpenAI's `gpt-3.5-turbo`**, with LangSmith tracing enabled. |
| `app.py` (LangServe API) | `api/` | FastAPI + **LangServe** server exposing three chains as REST endpoints: `/openai` (raw ChatOpenAI), `/essay` (100-word essay generator via OpenAI), and `/poem` (children's poem generator via local Llama 2). |
| `client.py` | `api/` | Streamlit front-end that calls the LangServe `/essay` and `/poem` endpoints above. |
| `app.py` (Groq RAG demo) | `rag/` | Streamlit RAG app: loads a webpage (LangSmith docs), embeds it with Ollama's `nomic-embed-text`, indexes it in **FAISS**, and answers questions using **ChatGroq**. |
| `simplerag.ipynb` | `rag/` | 📝 Empty notebook — scaffold for a basic RAG walkthrough (likely using `attention.pdf` / `speech.txt` as source documents). |
| `agents.ipynb` | `agents/` | 📝 Empty notebook — scaffold for LangChain agent experiments. |
| `attention.pdf` | `data/` | Sample PDF ("Attention Is All You Need") for testing PDF ingestion/RAG. |
| `speech.txt` | `data/` | Sample plain-text document (Wilson's WWI address to Congress) for testing text ingestion/RAG. |
| `main.py` | root | Placeholder entry point for the eventual unified assistant. |


## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangChain, LangChain Community, LangChain Core |
| LLM Providers | Groq, OpenAI, Ollama (local Llama 2) |
| Embeddings | Ollama (`nomic-embed-text`), HuggingFace (`sentence-transformers`) |
| Vector Stores | FAISS, ChromaDB, Cassandra (via CassIO) |
| Retrieval Sources | Web pages, PDFs, Wikipedia, arXiv |
| Document Parsing | PyPDF, PyPDF2, BeautifulSoup4 |
| API Serving | FastAPI, LangServe, SSE-Starlette, Uvicorn |
| Frontend | Streamlit |
| Tracing | LangSmith |

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- [Ollama](https://ollama.com/) installed locally, with the `llama2` and `nomic-embed-text` models pulled, for the scripts that use local models:
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
OPENAI_API_KEY="your-openai-api-key"
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

### 2. OpenAI chatbot
```bash
streamlit run chatbot/app.py
```
Requires `OPENAI_API_KEY` in `.env`.

### 3. LangServe API + client
Start the API server:
```bash
python api/app.py
```
This serves endpoints at `http://localhost:8000` (`/openai`, `/essay`, `/poem`). In a separate terminal, launch the client UI:
```bash
streamlit run api/client.py
```

### 4. Groq-powered RAG demo
```bash
streamlit run rag/app.py
```
Requires `GROQ_API_KEY` in `.env` and Ollama running locally (for embeddings). Indexes the LangSmith documentation site and answers questions grounded in it.

### 5. Notebooks
Open `rag/simplerag.ipynb` or `agents/agents.ipynb` in Jupyter/VS Code to experiment interactively. Both are currently empty starting points.

## Sample Data

- `data/attention.pdf` — the "Attention Is All You Need" paper, useful for testing PDF-based RAG pipelines.
- `data/speech.txt` — a plain-text speech, useful for testing simple text ingestion and chunking.

