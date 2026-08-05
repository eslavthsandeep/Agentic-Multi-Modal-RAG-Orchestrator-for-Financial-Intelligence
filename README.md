# 🧠 OmniBrain — Agentic Multi-Modal RAG Orchestrator

> A production-grade system that ingests complex financial PDFs and answers analyst-style queries using a LangGraph supervisor that routes across specialized Search, SQL, and Vision agents. All answers are grounded with citations back to the source PDF page.

## Architecture

```text
       [ User Query ]
             │
             ▼
      [ Supervisor ] (LangGraph)
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
[Search]   [SQL]   [Vision]
(Qdrant) (SQLite) (GPT-4o)
    │        │        │
    └────────┼────────┘
             ▼
      [ Synthesizer ]
             │
             ▼
       [ Guardrails ]
             │
             ▼
     [ Final Answer ]
```

## Key Features

- **Agentic Orchestration**: LangGraph supervisor dynamically routes queries to specialized agents
- **Multi-Modal Retrieval**: Text chunks via OpenAI embeddings + chart/table images via CLIP embeddings in Qdrant
- **Cross-Modal Search**: Text queries can find relevant charts using CLIP's shared text-image embedding space
- **Self-Correcting RAG**: Automatic query rewriting when retrieved context has low relevance
- **SQL Agent**: Natural language to SQL for historical stock data, with query validation (SELECT-only guard)
- **Vision Analysis**: GPT-4o vision extracts numerical data from charts and graphs
- **Guardrails**: NeMo Guardrails blocks off-topic questions and ungrounded answers
- **Observability**: Langfuse tracing for token usage, latency, and execution paths
- **Async Ingestion**: Non-blocking PDF processing with status polling

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph + Command API) |
| LLM | GPT-4o (reasoning, vision, SQL gen) |
| Vector DB | Qdrant (Docker) |
| Text Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Image Embeddings | CLIP ViT-B/32 (512-dim) |
| SQL Database | SQLite |
| Backend | FastAPI (async) |
| Frontend | Streamlit |
| Guardrails | NeMo Guardrails |
| Observability | Langfuse |
| PDF Processing | PyMuPDF + pdfplumber |

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key with GPT-4o access

### 1. Clone and setup
```bash
git clone <repo-url>
cd omnibrain
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Qdrant
```bash
docker-compose up -d
```

### 3. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Seed stock data
```bash
python -m app.db.seed_stock_data
```

### 5. Add a test PDF
Download a 10-K annual report from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=include&count=10) (Apple Inc.) and place it in `backend/data/sample_pdfs/`.

### 6. Run the backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Run the frontend
```bash
cd ../frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 8. Run tests
```bash
cd ../backend
python -m pytest tests/ -v
```

## API Documentation

Once running, visit http://localhost:8000/docs for the interactive Swagger UI.

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload a PDF for processing |
| GET | `/api/upload/{id}/status` | Check processing status |
| POST | `/api/query` | Query the processed document |
| GET | `/api/agents/status` | System health check |

## Project Structure

```text
omnibrain/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph nodes and supervisor logic
│   │   ├── db/              # SQLite setup and seeding
│   │   ├── ingestion/       # PDF parsing and embedding generation
│   │   ├── models/          # Pydantic schemas
│   │   ├── routers/         # FastAPI endpoints
│   │   └── main.py          # FastAPI application entry
│   ├── tests/               # Pytest suite
│   │   ├── conftest.py
│   │   ├── test_agents.py
│   │   ├── test_api.py
│   │   ├── test_guardrails.py
│   │   └── test_ingestion.py
│   ├── requirements.txt
│   └── data/                # Data storage (pdfs, db)
├── frontend/
│   ├── components/          # Streamlit custom UI elements
│   │   ├── agent_trace_panel.py
│   │   ├── chat_ui.py
│   │   └── citation_viewer.py
│   ├── streamlit_app.py     # Main Streamlit dashboard
│   └── requirements.txt
├── docker-compose.yml       # Qdrant and other services
├── .env.example
└── README.md
```

## Design Decisions

### Why LangGraph over LangChain AgentExecutor?
LangGraph provides explicit state machine control — you see exactly which path a query takes, can add conditional retry loops, and debug routing decisions. AgentExecutor is a black box.

### Why separate text and image Qdrant collections?
Different embedding dimensions (1536 for OpenAI text vs 512 for CLIP images). Separate collections allow independent tuning of search parameters.

### Why CLIP text encoder for image search?
CLIP's text and image encoders share a latent space, so text queries embedded with CLIP's text encoder can be compared against CLIP image embeddings. Using OpenAI's text embeddings (1536-dim) against CLIP image embeddings (512-dim) would be a dimension mismatch and semantically meaningless.

### Why SQL query validation?
Even though GPT-4o generates the SQL, defense-in-depth means never trusting unvalidated input. The SELECT-only guard prevents any accidental or adversarial data modification.

### Why async ingestion?
PDF parsing + CLIP loading + embedding generation are blocking operations. Running them synchronously in a FastAPI endpoint would stall the event loop. BackgroundTasks keeps the API responsive.

## License

MIT
