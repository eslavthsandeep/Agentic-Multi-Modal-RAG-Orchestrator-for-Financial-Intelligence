# 🧠 OmniBrain — Agentic Multi-Modal RAG Orchestrator for Financial Intelligence

> A production-grade financial intelligence system that ingests complex corporate financial PDFs (such as SEC 10-K annual reports) and synthesizes analyst-grade answers using a **LangGraph State Machine Supervisor** that dynamically orchestrates specialized **Search**, **SQL**, and **Vision** agents. All answers are strictly grounded with inline page-level citations.

---

## 📐 Architecture & Multi-Agent Flow

```text
                                 ┌───────────────────────────┐
                                 │   User Query & Document   │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │  NeMo Guardrails Check    │
                                 └─────────────┬─────────────┘
                                               │ (Passed)
                                               ▼
                                 ┌───────────────────────────┐
                                 │  Supervisor Node (GPT-4o) │
                                 └───────┬─────┬─────┬───────┘
                                         │     │     │
                 ┌───────────────────────┘     │     └───────────────────────┐
                 ▼                             ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
   │       Search Agent        │ │         SQL Agent         │ │       Vision Agent        │
   │  - Qdrant Text (1536-dim) │ │ - Text-to-SQL (SQLite)    │ │ - GPT-4o Multimodal Vision│
   │  - CLIP Images (512-dim)  │ │ - SELECT-only Validation  │ │ - Base64 Image Processing │
   │  - Hybrid Reranker (BM25) │ │ - Stock History Database  │ │ - Qdrant Cross-Lookup     │
   │  - Self-RAG Rewrite Loop  │ └─────────────┬─────────────┘ └─────────────┬─────────────┘
   └─────────────┬─────────────┘               │                             │
                 │                             │                             │
                 └─────────────────────────────┼─────────────────────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │      Synthesizer Node     │
                                 │ - Multi-Source Aggregator │
                                 │ - Page-Level Citations    │
                                 │ - Intelligent Fallback    │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │     Final Answer + Trace  │
                                 └───────────────────────────┘
```

---

## 🌟 Key Features & Core Capabilities

- **Agentic Orchestration**: LangGraph State Machine control flow providing explicit step-by-step routing visibility instead of black-box LLM chains.
- **Multi-Modal Vector Retrieval**:
  - **Text Chunks**: OpenAI `text-embedding-3-small` (1536-dimensional vectors) in Qdrant `omnibrain_text`.
  - **Image Elements**: OpenAI CLIP `clip-vit-base-patch32` (512-dimensional vectors) in Qdrant `omnibrain_images`.
- **Cross-Modal Search**: Enables text queries to retrieve chart and table images by aligning text query embeddings with CLIP's shared text-image latent space.
- **Hybrid Reranking**: Combines dense vector similarity with dynamic n-gram phrase matching and BM25 term weighting to ensure financial table entries and executive disclosures rank #1.
- **Self-Correcting RAG**: Evaluates top retrieval scores against a relevance threshold (`0.65`); if relevance is low, automatically rewrites the query using GPT-4o for up to `2` correction attempts.
- **SQL Financial Agent**: Translates natural language questions about historical stock prices, volume, and market trends into SQL with an explicit **SELECT-only security guard** (`_validate_sql`).
- **Multimodal Vision Analysis**: Utilizes GPT-4o vision capabilities to decode extracted chart images, trends, and visual table data.
- **NeMo Guardrails**: Pre-execution input validation layer that blocks off-topic queries and ungrounded responses.
- **Langfuse Observability**: Embedded `@traced` instrumentation logging latency, token consumption, and execution step traces.
- **Fault-Tolerant Fallback Engine**: Built-in intelligent fallback handlers for demo resilience when external API quotas are exceeded.

---

## 📂 Repository Directory Structure

```text
Omnibrain-project/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── search_agent.py     # Qdrant text/image retrieval, hybrid reranking, Self-RAG loop
│   │   │   ├── sql_agent.py        # Text-to-SQL generation, SELECT security guard, SQLite execution
│   │   │   ├── state.py            # AgentState TypedDict definition for LangGraph state machine
│   │   │   ├── supervisor.py       # LangGraph graph builder, supervisor router node, synthesizer node
│   │   │   └── vision_agent.py     # Base64 image encoding, GPT-4o vision analysis, Qdrant fallback
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_query.py     # POST /api/query endpoint
│   │   │   ├── routes_status.py    # GET /api/agents/status health endpoint
│   │   │   └── routes_upload.py    # POST /api/upload and GET /api/upload/{id}/status endpoints
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── qdrant_client.py    # Qdrant client singleton, collection management, vector search
│   │   │   ├── seed_stock_data.py  # Seeding 1,255 daily stock records into SQLite
│   │   │   └── sql_client.py       # SQLite connection pool, schema initialization, query execution
│   │   ├── guardrails/
│   │   │   ├── __init__.py
│   │   │   ├── guard.py            # NeMo Guardrails input verification interface
│   │   │   └── nemo_config/        # YAML/Colang rails configuration files
│   │   │       ├── config.yml
│   │   │       ├── prompts.yml
│   │   │       └── rails.co
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py          # Recursive character text splitting (500 size, 50 overlap)
│   │   │   ├── embedder.py         # OpenAI text embeddings & HuggingFace CLIP image embeddings
│   │   │   └── pdf_parser.py       # PyMuPDF (fitz) text and chart image extraction
│   │   ├── observability/
│   │   │   ├── __init__.py
│   │   │   └── langfuse_client.py  # Langfuse tracing decorator and client wrapper
│   │   ├── config.py               # Pydantic BaseSettings application configuration
│   │   └── main.py                 # FastAPI application entry point, CORS, and lifecycle setup
│   ├── data/                       # Runtime storage (SQLite stock.db, qdrant_db, uploads, images)
│   ├── tests/                      # Pytest suite
│   │   ├── conftest.py
│   │   ├── test_agents.py
│   │   ├── test_api.py
│   │   ├── test_guardrails.py
│   │   └── test_ingestion.py
│   ├── Dockerfile                  # Container definition for backend FastAPI server
│   ├── requirements.txt            # Backend Python dependencies
│   ├── debug_scores.py             # Diagnostic script for retrieval score inspection
│   └── verify_all_4_bugs.py        # Comprehensive verification test suite
├── frontend/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── agent_trace_panel.py    # Streamlit visualizer for LangGraph execution trace steps
│   │   ├── chat_ui.py              # Interactive multi-modal chat interface
│   │   └── citation_viewer.py     # Interactive citation and snippet preview drawer
│   ├── streamlit_app.py            # Main Streamlit dashboard application
│   └── requirements.txt            # Frontend Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules for secrets, DBs, and virtual environments
├── docker-compose.yml              # Docker services setup (Qdrant Vector DB)
└── README.md                       # Complete system documentation
```

---

## 🗄️ Database Schemas & Data Storage

### 1. Vector Collections (Qdrant)

#### Collection: `omnibrain_text`
- **Vector Dimension**: `1536` (OpenAI `text-embedding-3-small`)
- **Distance Metric**: `Cosine`
- **Payload Schema**:
  ```json
  {
    "document_id": "string (UUID or custom doc key)",
    "page_num": "integer (1-indexed page number)",
    "chunk_index": "integer (positional index in page)",
    "text": "string (extracted chunk content)",
    "source_type": "string ('text' or 'table')"
  }
  ```

#### Collection: `omnibrain_images`
- **Vector Dimension**: `512` (CLIP `openai/clip-vit-base-patch32`)
- **Distance Metric**: `Cosine`
- **Payload Schema**:
  ```json
  {
    "document_id": "string",
    "page_num": "integer",
    "image_path": "string (filesystem path to extracted image)"
  }
  ```

### 2. Relational Database (SQLite `stock.db`)

#### Table: `stock_history`
Stores daily historical market records (pre-seeded with 1,255 AAPL trading days):
```sql
CREATE TABLE IF NOT EXISTS stock_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    close_price REAL NOT NULL,
    volume INTEGER NOT NULL
);
```

---

## 🔄 Agent Control Flow & State Machine

The orchestration pipeline is built using LangGraph's `StateGraph` around the shared `AgentState` schema:

```python
class AgentState(TypedDict):
    query: str
    document_id: str
    chat_history: list[dict]
    messages: list[dict]
    route_decision: str | None
    search_results: list[dict] | None
    sql_results: list[dict] | None
    vision_results: list[dict] | None
    self_correction_attempts: int
    final_answer: str | None
    citations: list[dict]
    agent_trace: list[dict]
    guardrail_flag: str | None
```

### Routing Logic (`supervisor_node`)

The **Supervisor Node** inspects the user query and outputs a JSON classification decision:

| Route Decision | Destination Node | Intended Target Queries |
| :--- | :--- | :--- |
| `"search"` | `search_agent` | Conceptual text, ESG/sustainability, executive names, risk factors |
| `"sql"` | `sql_agent` | Stock prices, trading volume, average closing price, peak dates |
| `"vision"` | `vision_agent` | Segment revenue charts, visual diagrams, graphic figures |
| `"multi"` | `search_agent` $\rightarrow$ `sql_agent` | Combined questions requiring both PDF text and stock SQL data |
| `"end"` | `synthesizer` | Off-topic or unanswerable requests |

---

## 🛠️ Hybrid Reranking & Retrieval Optimization

In `search_agent.py`, the system implements a hybrid reranker (`_hybrid_rerank`) that combines vector cosine similarity scores with dynamic n-gram phrase matching:

$$\text{Final Score} = \min\left(1.0,\, \text{Base Score} \times 0.2 + \text{Phrase Boost} + \text{Term Boost}\right)$$

Where:
- **Dynamic N-Grams**: Automatically extracts 2-gram and 3-gram phrases from the query (e.g., `"effective tax rate"`, `"supply chain"`, `"chief executive officer"`).
- **Term Boost**: Calculates the ratio of matching non-stopword query terms present in the candidate chunk.
- **Boost Factor**: Gives a `0.25` score boost per phrase match and up to `0.40` for full term coverage.

---

## 🌐 API Reference

### Base URL: `http://localhost:8000`
Interactive Swagger UI documentation is available at `http://localhost:8000/docs`.

### 1. Document Upload
* **Endpoint**: `POST /api/upload`
* **Content-Type**: `multipart/form-data`
* **Request**: `file` (PDF document)
* **Response**:
  ```json
  {
    "document_id": "f1922693-d5f",
    "filename": "apple_10k_2025.pdf",
    "status": "processing"
  }
  ```

### 2. Upload Status Check
* **Endpoint**: `GET /api/upload/{document_id}/status`
* **Response**:
  ```json
  {
    "document_id": "f1922693-d5f",
    "status": "completed",
    "pages_processed": 64,
    "chunks_created": 312
  }
  ```

### 3. Query Document
* **Endpoint**: `POST /api/query`
* **Content-Type**: `application/json`
* **Request Payload**:
  ```json
  {
    "document_id": "f1922693-d5f",
    "query": "What was Apple's total net sales in fiscal 2025, and how does that compare to the stock price trend?",
    "chat_history": []
  }
  ```
* **Response Payload**:
  ```json
  {
    "answer": "### 📊 Multi-Agent Analysis: Net Sales vs. Stock Price Trend\n\n#### 1. Document Financial Performance (Search Agent)\n- **Fiscal 2025 Total Net Sales:** Apple reported total net sales of **$391,035 million** [Page 32, text].\n\n#### 2. Stock Market Performance (SQL Agent)\n- **AAPL Stock Price Trend:** AAPL traded between **$224.23 and $235.86** [SQL query: `SELECT date, close_price, volume FROM stock_history`].\n\n*(Synthesized via OmniBrain Multi-Modal Orchestrator)*",
    "route_taken": ["multi"],
    "citations": [
      {
        "page": 32,
        "source_type": "text",
        "snippet": "Total net sales 391,035..."
      }
    ],
    "agent_trace": [
      {"step": "supervisor", "detail": "Routed to multi"},
      {"step": "search", "detail": "Retrieved 5 items"},
      {"step": "sql", "detail": "Executed SQL query"},
      {"step": "synthesizer", "detail": "Produced answer"}
    ],
    "guardrail_status": "passed"
  }
  ```

### 4. Health Check & Status
* **Endpoint**: `GET /api/agents/status`
* **Response**:
  ```json
  {
    "status": "healthy",
    "qdrant_connected": true,
    "sqlite_connected": true,
    "active_agents": ["supervisor", "search_agent", "sql_agent", "vision_agent", "synthesizer"]
  }
  ```

---

## ⚡ Quick Start & Setup Guide

### 1. Prerequisites
- **Python**: `3.10` or `3.11`
- **Docker**: (Optional, for containerized Qdrant instance)
- **API Keys**: OpenAI API key with GPT-4o model access

### 2. Environment Configuration
Copy `.env.example` to `.env` in the root directory (and in `backend/.env`):
```bash
OPENAI_API_KEY=sk-proj-your-openai-api-key
QDRANT_URL=http://localhost:6333
DATABASE_URL=sqlite:///./data/stock.db
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start Infrastructure (Qdrant Vector DB)
Option A: Using Docker Compose
```bash
docker-compose up -d
```
Option B: Standalone / Embedded Mode
If Docker is not running, OmniBrain automatically falls back to local disk storage in `backend/data/qdrant_db`.

### 4. Seed Stock Market Data
Seed the SQLite database with 1,255 historical market records:
```bash
cd backend
python -m app.db.seed_stock_data
```

### 5. Launch Backend FastAPI Server
```bash
cd backend
python -u -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 6. Launch Frontend Streamlit App
In a separate terminal window:
```bash
cd frontend
streamlit run streamlit_app.py --server.port 8501
```

Access the UI in your web browser at: **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Testing & Verification Suite

Run the full automated test suite:
```bash
cd backend
python -m pytest tests/ -v
```

Run comprehensive end-to-end verification for all query routes:
```bash
cd backend
python verify_all_4_bugs.py
```

---

## 🛡️ Security & Guardrails

1. **SQL Injection Defense**: The SQL agent enforces a strict pre-execution AST validator (`_validate_sql`) allowing **ONLY SELECT** queries. Any statement containing `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, or `TRUNCATE` is rejected before hitting the database.
2. **NeMo Input Guardrails**: Intercepts off-topic queries and malicious prompt injections before sending requests to the LLM orchestration graph.
3. **Strict Citation Grounding**: The synthesizer prompt instructs the model to rely solely on context blocks, outputting inline citations `[Page X, <type>]`. When context is missing, it explicitly returns *"I couldn't find this information in the retrieved document context."*

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
