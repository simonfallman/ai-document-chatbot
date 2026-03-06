# AI Document Chatbot

A RAG-based chatbot that lets you upload a document and ask questions about its content. Built with Python, LangChain, AWS Bedrock, and ChromaDB.

## Stack

- **Streamlit** — web UI
- **LangChain** — RAG orchestration
- **AWS Bedrock** — embeddings (Amazon Titan `titan-embed-text-v2`) + LLM (Meta Llama 3 / Claude 3 Haiku)
- **ChromaDB** — vector store, persisted to disk with per-document isolation
- **SQLite** — persistent chat history and conversation management
- **Docker** — containerization
- **GitHub Actions** — CI/CD pipeline (test + auto-deploy)

## Features

- Upload PDF, TXT, or DOCX files (max 5MB)
- Semantic search over document content using vector embeddings
- Conversational memory — follow-up questions work correctly
- Persistent chat history with conversation sidebar
- Per-document vector isolation — each conversation remembers its own document
- Streaming responses
- Password protection (optional)
- **Tools:** document summarization (map-reduce) and FAQ generation

## Local Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
```
Edit `.env` and set your AWS credentials. Optionally set `APP_PASSWORD` to enable password protection.

**3. Run**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker

```bash
docker build -t ai-document-chatbot .
docker run -d -p 8501:8501 -v $(pwd)/chroma_db:/app/chroma_db --env-file .env --restart unless-stopped ai-document-chatbot
```

The `-v` flag mounts `chroma_db` from the host so indexed documents and chat history survive container restarts.

**Access the app** at `http://<your-server-ip>:8501`

## CI/CD

Every push to `main` triggers a GitHub Actions pipeline that:
1. Runs the test suite
2. SSHs into the server and redeploys if tests pass

## Configuration

| Setting | Default | Notes |
|---|---|---|
| LLM | `meta.llama3-8b-instruct-v1:0` | Change in `build_chain()` in `app.py` |
| Embedding model | `amazon.titan-embed-text-v2:0` | Change in `get_embeddings()` |
| Chunk size | 500 | Change in `build_vectorstore()` |
| Chunk overlap | 50 | Change in `build_vectorstore()` |
| Max file size | 5MB | Change `MAX_FILE_SIZE` in `app.py` |

## Running Tests

```bash
python3 -m pytest test_pipeline.py -v
```
