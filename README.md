# AI Document Chatbot

A RAG-based chatbot that lets you upload a document and ask questions about its content.

## Stack

- **Streamlit** — web UI
- **LangChain** — RAG orchestration
- **OpenAI** — embeddings (`text-embedding-3-small`) + chat (`gpt-3.5-turbo`)
- **ChromaDB** — vector store, persisted to disk

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure API key**
```bash
cp .env.example .env
```
Edit `.env` and set your `OPENAI_API_KEY`.

**3. Run**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

1. Upload a PDF, TXT, or DOCX file (max 5MB) using the sidebar
2. Wait for the document to be indexed (only happens once per file)
3. Ask questions in the chat input
4. Expand **Sources** under any answer to see the retrieved chunks

## Docker

```bash
docker build -t ai-document-chatbot .
docker run -d -p 8501:8501 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  --restart unless-stopped \
  ai-document-chatbot
```

- `-d` runs the container in the background
- `--restart unless-stopped` ensures the app restarts automatically after a server reboot
- The `chroma_db` volume mount ensures indexed documents survive container restarts

**Access the app** at `http://<your-server-ip>:8501`

## Configuration

| Setting | Default | Notes |
|---|---|---|
| LLM | `gpt-3.5-turbo` | Change in `build_chain()` in `app.py` |
| Embedding model | `text-embedding-3-small` | Change in `build_vectorstore()` |
| Chunk size | 500 | Change in `build_vectorstore()` |
| Chunk overlap | 50 | Change in `build_vectorstore()` |
| Max file size | 5MB | Change `MAX_FILE_SIZE` in `app.py` |

## Running Tests

```bash
python3 -m pytest test_pipeline.py
```
