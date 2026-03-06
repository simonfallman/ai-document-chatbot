import os
import hashlib
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LCDocument
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
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

Path(DOCUMENTS_DIR).mkdir(exist_ok=True)
Path(CHROMA_DIR).mkdir(exist_ok=True)


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


def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Step 1: condense question + history into a standalone question
    condense_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("human", "Rephrase the above as a standalone question, preserving all context."),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    # Step 2: retrieve + answer
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using only the context below.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_and_answer(inp):
        standalone = condense_chain.invoke(inp) if inp.get("chat_history") else inp["input"]
        docs = retriever.invoke(standalone)
        context = format_docs(docs)
        answer = (qa_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": inp.get("chat_history", []),
            "input": inp["input"],
        })
        return {"answer": answer, "context": docs}

    chain = RunnableWithMessageHistory(
        RunnableLambda(retrieve_and_answer),
        lambda session_id: st.session_state.chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return chain


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Document Chatbot", page_icon="📄")
st.title("📄 AI Document Chatbot")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Create a .env file with your key.")
    st.stop()

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Sidebar – file upload
with st.sidebar:
    st.header("Upload Document")
    supported = ["pdf", "txt"] + (["docx"] if DOCX_SUPPORTED else [])
    uploaded = st.file_uploader(f"Supported: {', '.join(supported)}", type=supported)

    if uploaded:
        if uploaded.size > MAX_FILE_SIZE:
            st.error("File too large. Please upload a file under 5MB.")
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
                        vectorstore = build_vectorstore(docs)
                        indexed[fhash] = uploaded.name
                        save_indexed_hashes(indexed)
                        st.session_state.chain = build_chain(vectorstore)
                        st.session_state.messages = []
                        st.session_state.chat_history = ChatMessageHistory()
                        st.success(f"Indexed: {uploaded.name}")
            else:
                if st.session_state.chain is None:
                    vectorstore = get_vectorstore()
                    st.session_state.chain = build_chain(vectorstore)
                st.info(f"Already indexed: {indexed[fhash]}")

# Chat area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(src)

if prompt := st.chat_input("Ask a question about your document..."):
    if st.session_state.chain is None:
        st.warning("Please upload a document first.")
    else:
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
                sources = [
                    doc.page_content[:200] for doc in result.get("context", [])
                ]

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.caption(src)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
