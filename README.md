# AI Document Chatbot

A production RAG chatbot for semantic document Q&A. Upload PDFs, Word documents, or text files and ask questions across all of them simultaneously. Built with Python, LangChain, AWS Bedrock, and ChromaDB. Live at [simonfallman.xyz](https://simonfallman.xyz).

![App screenshot](screenshot.png)

## Architecture

```mermaid
flowchart TD
    User([User]) -->|uploads file| Ingest
    User -->|asks question| Intent

    subgraph Ingest["Ingest Pipeline"]
        A["LangChain Loader<br/>PDF / TXT / DOCX"] --> B["RecursiveCharacterTextSplitter<br/>chunk_size=500, overlap=50"]
        B --> C["Stamp metadata<br/>document_name · collection_hash · chunk_index"]
        C --> D["BedrockEmbeddings<br/>Amazon Titan"]
        D --> E[("ChromaDB<br/>per-document collection<br/>keyed by MD5 hash")]
    end

    subgraph Retrieve["Retrieval & Generation"]
        Intent{"Intent<br/>Detection"} -->|summarize / faq| Tools["Map-Reduce Tools<br/>Summarize · FAQ"]
        Intent -->|normal question| Condense["Condense question<br/>with chat history"]
        Condense --> Multi["multi_retrieve<br/>query each collection<br/>merge · rank · deduplicate"]
        Multi --> E
        E --> Multi
        Multi --> Context["Top-k chunks<br/>as context"]
        Context --> LLM["Claude 3.5 Haiku<br/>via AWS Bedrock"]
        Tools --> LLM
        LLM --> Answer([Answer + Sources])
    end

    subgraph Persist["Persistence"]
        Answer --> SQLite[("SQLite<br/>conversations · messages<br/>collection_hashes")]
    end
```

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG orchestration | LangChain |
| LLM | Claude 3.5 Haiku via AWS Bedrock |
| Embeddings | Amazon Titan `titan-embed-text-v2` via AWS Bedrock |
| Vector store | ChromaDB (persisted, per-document collections) |
| Chat history | SQLite |
| Observability | Prometheus + Grafana |
| Experiment tracking | MLflow |
| Containerization | Docker Compose |
| CI/CD | GitHub Actions |

## Features

- Upload PDF, TXT, or DOCX files (max 5MB)
- **Multi-source retrieval** — select multiple documents and ask questions across all of them simultaneously
- Semantic search via vector embeddings — retrieves by meaning, not keywords
- Results merged and ranked by relevance score across all active documents
- Chunk metadata stamped at index time (`document_name`, `collection_hash`, `chunk_index`) for guaranteed provenance
- Conversational memory — follow-up questions correctly condense chat history into standalone queries
- Persistent chat history with conversation sidebar
- Per-document vector isolation via MD5-keyed ChromaDB collections
- **Tools:** multi-source summarization (map-reduce) and FAQ generation via intent detection
- Password protection (optional, via `APP_PASSWORD`)
- **Observability:** Prometheus histograms track end-to-end and retrieval latency separately — Grafana dashboards show real-time query volume, latency distribution, error rate, and upload counts
- **Experiment tracking:** MLflow logs every query as a run with config params (chunk size, k, model) alongside quality metrics (relevance score, latency, chunks retrieved) — making configuration changes measurable and comparable across runs

## Local Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
```
Edit `.env` and add your AWS credentials. Optionally set `APP_PASSWORD` to enable password protection.

**3. Run**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker Compose (full stack)

Runs the app alongside MLflow, Prometheus, and Grafana:

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| Chatbot | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Observability & Experiment Tracking

RAG quality is invisible without instrumentation. This stack makes every query measurable and every configuration change comparable.

### Real-Time Metrics — Prometheus + Grafana

Five metrics are tracked on every request:

| Metric | Type | What it tells you |
|---|---|---|
| `query_total` | Counter | Request volume and traffic patterns |
| `query_latency_seconds` | Histogram | End-to-end latency distribution (p50, p95, p99) |
| `retrieval_latency_seconds` | Histogram | ChromaDB retrieval time in isolation |
| `document_uploads_total` | Counter | Ingestion volume |
| `errors_total{type}` | Counter | Failure rate, labelled by exception class |

Latency is tracked as a **histogram** — not just an average. This surfaces whether slowness is consistent or spiky, which matters when debugging retrieval at scale.

Grafana dashboards at `http://localhost:3000` show all of these in real time. If retrieval latency spikes after a chunk size change, you see it immediately.

### Experiment Tracking — MLflow

Every query is logged as an MLflow run with both **config params** and **quality metrics**:

```python
# Config — what was the system set to?
mlflow.log_param("chunk_size", 500)
mlflow.log_param("chunk_overlap", 50)
mlflow.log_param("model_id", MODEL_ID)
mlflow.log_param("k", 6)

# Quality — how did it perform?
mlflow.log_metric("retrieval_latency_ms", ...)
mlflow.log_metric("avg_relevance_score", ...)
mlflow.log_metric("num_chunks_retrieved", ...)
mlflow.log_metric("total_latency_ms", ...)
mlflow.set_tag("query_type", "rag" | "summarize" | "faq")
```

This makes configuration changes testable. Change `chunk_size` from 500 to 250, run a set of queries, and compare runs in the MLflow UI — did average relevance scores go up? Did latency drop? The data decides, not intuition.

MLflow at `http://localhost:5000`.

## CI/CD

Every push to `main` triggers a GitHub Actions pipeline that:
1. Runs the full test suite (24 tests)
2. SSHs into the server and redeploys if all tests pass
3. Health checks `https://simonfallman.xyz` to confirm the app is live

## Configuration

| Setting | Default | Notes |
|---|---|---|
| LLM | `claude-3-5-haiku` (Bedrock inference profile) | Change `MODEL_ID` in `app.py` |
| Embedding model | `amazon.titan-embed-text-v2:0` | Change in `get_embeddings()` |
| Chunk size | 500 | Change `CHUNK_SIZE` in `app.py` |
| Chunk overlap | 50 | Change `CHUNK_OVERLAP` in `app.py` |
| Retrieval k | 6 | Change `RETRIEVAL_K` in `app.py` |
| Max file size | 5 MB | Change `MAX_FILE_SIZE` in `app.py` |

## Running Tests

```bash
python3 -m pytest test_pipeline.py -v
```
