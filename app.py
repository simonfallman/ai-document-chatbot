import os
import hashlib
import json
import sqlite3
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_SUPPORTED = True
except ImportError:
    DOCX_SUPPORTED = False

load_dotenv()

CHROMA_DIR = "./chroma_db"
DOCUMENTS_DIR = "./documents"
HASH_STORE = "./chroma_db/indexed_files.json"
DB_PATH = "./chroma_db/chat_history.db"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

Path(DOCUMENTS_DIR).mkdir(exist_ok=True)
Path(CHROMA_DIR).mkdir(exist_ok=True)


# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            document TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL REFERENCES conversations(id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # migrate existing DBs that don't have the document column
    cols = [r[1] for r in con.execute("PRAGMA table_info(conversations)").fetchall()]
    if "document" not in cols:
        con.execute("ALTER TABLE conversations ADD COLUMN document TEXT")
    con.commit()
    con.close()


def create_conversation(title: str, document: str) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute("INSERT INTO conversations (title, document) VALUES (?, ?)", (title, document))
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def load_conversations() -> list:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, title, document FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [{"id": r[0], "title": r[1], "document": r[2]} for r in rows]


def delete_conversation(cid: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
    con.execute("DELETE FROM conversations WHERE id = ?", (cid,))
    con.commit()
    con.close()


def save_message(cid: int, role: str, content: str, sources: list = None):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (cid, role, content, json.dumps(sources) if sources else None)
    )
    con.commit()
    con.close()


def load_messages(cid: int) -> list:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY id",
        (cid,)
    ).fetchall()
    con.close()
    return [
        {"role": r, "content": c, "sources": json.loads(s) if s else None}
        for r, c, s in rows
    ]


init_db()


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def load_indexed_hashes() -> dict:
    if Path(HASH_STORE).exists():
        with open(HASH_STORE) as f:
            return json.load(f)
    return {}


def save_indexed_hashes(hashes: dict):
    with open(HASH_STORE, "w") as f:
        json.dump(hashes, f)


def load_document(path: str, suffix: str):
    if suffix == ".pdf":
        return PyPDFLoader(path).load()
    elif suffix == ".txt":
        return TextLoader(path).load()
    elif suffix == ".docx" and DOCX_SUPPORTED:
        return Docx2txtLoader(path).load()
    return None


def build_vectorstore(docs, collection_name: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def get_vectorstore(collection_name: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── Tools ─────────────────────────────────────────────────────────────────────

SUMMARIZE_TRIGGERS = re.compile(
    r"\b(summarize|summary|summarise|overview|tldr|tl;dr|what is this (document|file) about)\b",
    re.IGNORECASE,
)

FAQ_TRIGGERS = re.compile(
    r"\b(faq|frequently asked|generate questions|what are the (key |common )?questions|quiz me)\b",
    re.IGNORECASE,
)


def tool_summarize(vectorstore) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    all_docs = vectorstore.get()["documents"]
    # Map: summarize each batch of 10 chunks
    batch_size = 10
    batch_summaries = []
    for i in range(0, len(all_docs), batch_size):
        batch = "\n\n".join(all_docs[i:i + batch_size])
        summary = llm.invoke(
            f"Summarize the following text concisely:\n\n{batch}"
        ).content
        batch_summaries.append(summary)
    # Reduce: summarize the summaries
    if len(batch_summaries) == 1:
        return batch_summaries[0]
    combined = "\n\n".join(batch_summaries)
    return llm.invoke(
        f"Combine these partial summaries into one coherent summary:\n\n{combined}"
    ).content


def tool_faq(vectorstore) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    all_docs = vectorstore.get()["documents"]
    # Take a representative sample of chunks
    sample = "\n\n".join(all_docs[:20])
    return llm.invoke(
        f"Based on the following document content, generate 5 frequently asked questions "
        f"and their answers. Format each as:\n\nQ: ...\n\nA: ...\n\n{sample}"
    ).content


def build_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    condense_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Rephrase the above as a standalone question, preserving all context."),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using only the context below.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_and_answer(inp):
        question = inp["input"]

        # Tool: summarize
        if SUMMARIZE_TRIGGERS.search(question):
            answer = tool_summarize(vectorstore)
            return {"answer": answer, "context": []}

        # Tool: FAQ
        if FAQ_TRIGGERS.search(question):
            answer = tool_faq(vectorstore)
            return {"answer": answer, "context": []}

        # Default: RAG
        standalone = condense_chain.invoke(inp) if inp.get("chat_history") else question
        docs = retriever.invoke(standalone)
        answer = (qa_prompt | llm | StrOutputParser()).invoke({
            "context": format_docs(docs),
            "chat_history": inp.get("chat_history", []),
            "input": question,
        })
        return {"answer": answer, "context": docs}

    return RunnableWithMessageHistory(
        RunnableLambda(retrieve_and_answer),
        lambda session_id: st.session_state.chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def switch_conversation(cid: int):
    msgs = load_messages(cid)
    history = ChatMessageHistory()
    for msg in msgs:
        if msg["role"] == "user":
            history.add_user_message(msg["content"])
        else:
            history.add_ai_message(msg["content"])
    st.session_state.current_conversation_id = cid
    st.session_state.messages = msgs
    st.session_state.chat_history = history


def new_conversation():
    st.session_state.current_conversation_id = None
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="wide")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Create a .env file with your key.")
    st.stop()

# Password protection
APP_PASSWORD = os.getenv("APP_PASSWORD")
if APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("📄 AI Document Chatbot")
        with st.form("login_form"):
            pwd = st.text_input("Enter password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# Session state defaults
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "active_document" not in st.session_state:
    st.session_state.active_document = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 AI Document Chatbot")

    # New chat button
    if st.button("＋  New chat", use_container_width=True):
        new_conversation()
        st.rerun()

    st.divider()

    # Document upload
    st.subheader("Upload Document")
    supported = ["pdf", "txt"] + (["docx"] if DOCX_SUPPORTED else [])
    uploaded = st.file_uploader(f"Supported: {', '.join(supported)}", type=supported)

    if uploaded:
        if uploaded.size > MAX_FILE_SIZE:
            st.error("File too large. Max 5MB.")
        else:
            file_bytes = uploaded.read()
            fhash = file_hash(file_bytes)
            indexed = load_indexed_hashes()

            if fhash not in indexed:
                save_path = Path(DOCUMENTS_DIR) / uploaded.name
                save_path.write_bytes(file_bytes)
                with st.spinner("Processing document..."):
                    suffix = Path(uploaded.name).suffix.lower()
                    docs = load_document(str(save_path), suffix)
                    if docs is None:
                        st.error("Could not load document.")
                    else:
                        vectorstore = build_vectorstore(docs, collection_name=fhash)
                        indexed[fhash] = uploaded.name
                        save_indexed_hashes(indexed)
                        st.session_state.active_collection = fhash
                        st.session_state.active_document = uploaded.name
                        st.session_state.chain = build_chain(vectorstore)
                        st.success(f"Indexed: {uploaded.name}")
            else:
                if st.session_state.chain is None or st.session_state.get("active_collection") != fhash:
                    vectorstore = get_vectorstore(collection_name=fhash)
                    st.session_state.active_collection = fhash
                    st.session_state.active_document = indexed[fhash]
                    st.session_state.chain = build_chain(vectorstore)
                st.info(f"Already indexed: {indexed[fhash]}")

    st.divider()

    # Tools panel
    with st.expander("🛠 Tools"):
        st.markdown("**Summarize**")
        st.caption("Trigger: 'summarize', 'overview', 'tldr'")
        st.caption("Fetches all document chunks and generates a structured summary using map-reduce.")
        st.markdown("**FAQ Generator**")
        st.caption("Trigger: 'faq', 'generate questions', 'quiz me'")
        st.caption("Generates 5 frequently asked questions and answers based on the document content.")

    st.divider()

    # Conversation history list
    st.subheader("Conversations")
    conversations = load_conversations()
    if not conversations:
        st.caption("No conversations yet.")
    for conv in conversations:
        is_active = st.session_state.current_conversation_id == conv["id"]
        doc_label = conv.get("document") or "Unknown document"
        label = f"{'▶  ' if is_active else ''}{conv['title']}"
        if st.button(label, key=f"conv_{conv['id']}", use_container_width=True, help=doc_label):
            switch_conversation(conv["id"])
            st.rerun()
        if is_active:
            st.caption(f"📎 {doc_label}")
            if st.button("Delete this conversation", key=f"del_{conv['id']}", use_container_width=True):
                delete_conversation(conv["id"])
                new_conversation()
                st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
active_doc = st.session_state.active_document
st.title("Chat")
if active_doc:
    st.caption(f"📎 {active_doc}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    if src.get("label"):
                        st.caption(f"**{src['label']}**")
                    st.caption(src["text"])

# Quick action buttons
if st.session_state.chain is not None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Summarize document", use_container_width=True):
            st.session_state.quick_prompt = "Summarize this document"
    with col2:
        if st.button("❓ Generate FAQ", use_container_width=True):
            st.session_state.quick_prompt = "Generate FAQ"

prompt_value = st.session_state.pop("quick_prompt", None)
placeholder_text = f"Ask a question about {active_doc}..." if active_doc else "Upload a document to start chatting..."
if prompt := (prompt_value or st.chat_input(placeholder_text)):
    if st.session_state.chain is None:
        st.warning("Please upload a document first.")
    else:
        # Create a new conversation on first message
        if st.session_state.current_conversation_id is None:
            title = prompt[:60] + ("..." if len(prompt) > 60 else "")
            cid = create_conversation(title, st.session_state.active_document)
            st.session_state.current_conversation_id = cid

        cid = st.session_state.current_conversation_id
        save_message(cid, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "default"}},
                )
                answer = result["answer"]
                sources = []
                for doc in result.get("context", []):
                    meta = doc.metadata
                    page = meta.get("page")
                    source = meta.get("source", "")
                    label = Path(source).name if source else ""
                    if page is not None:
                        label = f"{label} — page {int(page) + 1}" if label else f"Page {int(page) + 1}"
                    sources.append({"text": doc.page_content[:200], "label": label})

            # Stream answer word by word
            placeholder = st.empty()
            displayed = ""
            for word in answer.split(" "):
                displayed += word + " "
                placeholder.markdown(displayed + "▌")
            placeholder.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        if src["label"]:
                            st.caption(f"**{src['label']}**")
                        st.caption(src["text"])

        save_message(cid, "assistant", answer, sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
