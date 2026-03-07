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
from langchain_aws import ChatBedrock, BedrockEmbeddings
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
            collection_hash TEXT,
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
    # migrate existing DBs
    cols = [r[1] for r in con.execute("PRAGMA table_info(conversations)").fetchall()]
    if "document" not in cols:
        con.execute("ALTER TABLE conversations ADD COLUMN document TEXT")
    if "collection_hash" not in cols:
        con.execute("ALTER TABLE conversations ADD COLUMN collection_hash TEXT")
    if "collection_hashes" not in cols:
        con.execute("ALTER TABLE conversations ADD COLUMN collection_hashes TEXT")
    con.commit()
    con.close()


def create_conversation(title: str, document: str, collection_hash: str = None, collection_hashes: list = None) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO conversations (title, document, collection_hash, collection_hashes) VALUES (?, ?, ?, ?)",
        (title, document, collection_hash, json.dumps(collection_hashes) if collection_hashes else None)
    )
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def load_conversations() -> list:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, title, document, collection_hash, collection_hashes FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    con.close()
    return [
        {
            "id": r[0], "title": r[1], "document": r[2], "collection_hash": r[3],
            "collection_hashes": json.loads(r[4]) if r[4] else None,
        }
        for r in rows
    ]


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


def get_embeddings():
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def build_vectorstore(docs, collection_name: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def get_vectorstore(collection_name: str):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=collection_name,
    )


def get_vectorstores(collection_names: list):
    return [get_vectorstore(name) for name in collection_names]


def multi_retrieve(vectorstores: list, query: str, k: int = 6) -> list:
    """Query each vectorstore, merge results by relevance score, deduplicate."""
    all_results = []
    for vs in vectorstores:
        all_results.extend(vs.similarity_search_with_relevance_scores(query, k=4))
    all_results.sort(key=lambda x: x[1], reverse=True)
    seen, unique = set(), []
    for doc, _ in all_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
        if len(unique) >= k:
            break
    return unique


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


def tool_summarize(vectorstores: list) -> str:
    llm = ChatBedrock(model_id="meta.llama3-8b-instruct-v1:0", region_name=os.getenv("AWS_REGION", "us-east-1"), model_kwargs={"temperature": 0})
    all_docs = []
    for vs in vectorstores:
        all_docs.extend(vs.get()["documents"])
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


def tool_faq(vectorstores: list) -> str:
    llm = ChatBedrock(model_id="meta.llama3-8b-instruct-v1:0", region_name=os.getenv("AWS_REGION", "us-east-1"), model_kwargs={"temperature": 0})
    all_docs = []
    for vs in vectorstores:
        all_docs.extend(vs.get()["documents"])
    # Take a representative sample of chunks
    sample = "\n\n".join(all_docs[:20])
    return llm.invoke(
        f"Based on the following document content, generate 5 frequently asked questions "
        f"and their answers. Format each as:\n\n**Q:** ...\n\n**A:** ...\n\nSeparate each Q&A pair with a blank line. Do not put Q and A on the same line.\n\n{sample}"
    ).content


def build_chain(vectorstores: list):
    llm = ChatBedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": 0},
    )

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
            answer = tool_summarize(vectorstores)
            return {"answer": answer, "context": []}

        # Tool: FAQ
        if FAQ_TRIGGERS.search(question):
            answer = tool_faq(vectorstores)
            return {"answer": answer, "context": []}

        # Default: RAG across all sources
        standalone = condense_chain.invoke(inp) if inp.get("chat_history") else question
        docs = multi_retrieve(vectorstores, standalone)
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

def switch_conversation(conv: dict):
    cid = conv["id"]
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

    # Reload vectorstores for this conversation's documents
    chashes = conv.get("collection_hashes") or (
        [conv["collection_hash"]] if conv.get("collection_hash") else []
    )
    if chashes:
        vectorstores = get_vectorstores(chashes)
        st.session_state.active_collections = chashes
        st.session_state.active_documents = conv.get("document", "").split(", ")
        st.session_state.chain = build_chain(vectorstores)
    else:
        st.session_state.active_collections = []
        st.session_state.active_documents = []
        st.session_state.chain = None


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
if "active_collections" not in st.session_state:
    st.session_state.active_collections = []
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []

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
                        build_vectorstore(docs, collection_name=fhash)
                        indexed[fhash] = uploaded.name
                        save_indexed_hashes(indexed)
                        if fhash not in st.session_state.active_collections:
                            st.session_state.active_collections = st.session_state.active_collections + [fhash]
                            st.session_state.active_documents = st.session_state.active_documents + [uploaded.name]
                        st.success(f"Indexed: {uploaded.name}")
            else:
                if fhash not in st.session_state.active_collections:
                    st.session_state.active_collections = st.session_state.active_collections + [fhash]
                    st.session_state.active_documents = st.session_state.active_documents + [indexed[fhash]]
                st.info(f"Already indexed: {indexed[fhash]}")

    # Document selector — pick which indexed docs are active
    indexed = load_indexed_hashes()
    if indexed:
        st.subheader("Active Documents")
        all_hashes = list(indexed.keys())
        all_names = [indexed[h] for h in all_hashes]
        current_names = [indexed[h] for h in st.session_state.active_collections if h in indexed]
        selected_names = st.multiselect("Chat with:", all_names, default=current_names)
        selected_hashes = [all_hashes[all_names.index(n)] for n in selected_names]
        if sorted(selected_hashes) != sorted(st.session_state.active_collections):
            st.session_state.active_collections = selected_hashes
            st.session_state.active_documents = selected_names
            if selected_hashes:
                st.session_state.chain = build_chain(get_vectorstores(selected_hashes))
            else:
                st.session_state.chain = None
            st.rerun()

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
    else:
        for conv in conversations:
            is_active = st.session_state.current_conversation_id == conv["id"]
            doc_label = conv.get("document") or "Unknown document"
            label = f"{'▶  ' if is_active else ''}{conv['title']}"
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True, help=doc_label):
                switch_conversation(conv)
                st.rerun()
            if is_active:
                st.caption(f"📎 {doc_label}")
                if st.button("Delete this conversation", key=f"del_{conv['id']}", use_container_width=True):
                    delete_conversation(conv["id"])
                    new_conversation()
                    st.rerun()

        st.divider()
        if st.button("Delete all conversations", use_container_width=True):
            st.session_state.confirm_delete_all = True
        if st.session_state.get("confirm_delete_all"):
            st.warning("Are you sure? This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, delete all", use_container_width=True):
                    for conv in conversations:
                        delete_conversation(conv["id"])
                    new_conversation()
                    st.session_state.confirm_delete_all = False
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
# ── Main chat area ────────────────────────────────────────────────────────────
active_docs = st.session_state.get("active_documents", [])
st.title("💬 Chat")
if active_docs:
    st.caption("📎 " + " · ".join(active_docs))

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
placeholder_text = f"Ask about {' & '.join(active_docs)}..." if active_docs else "Upload a document to start chatting..."
if prompt := (prompt_value or st.chat_input(placeholder_text)):
    if st.session_state.chain is None:
        st.warning("Please upload a document first.")
    else:
        # Create a new conversation on first message
        if st.session_state.current_conversation_id is None:
            title = prompt[:60] + ("..." if len(prompt) > 60 else "")
            doc_label = ", ".join(st.session_state.active_documents)
            cid = create_conversation(title, doc_label, collection_hashes=st.session_state.active_collections)
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
