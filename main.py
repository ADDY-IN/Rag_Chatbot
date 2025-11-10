import os
import subprocess
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# ---------- CONFIG ----------
st.set_page_config(page_title="⚓ Naval RAG Chatbot", layout="wide")
st.title("⚓ Naval Intelligence Chatbot — Prototype")
st.sidebar.header("⚙️ Control Panel")

DATA_DIR = "data"
PERSIST_DIR = "vectorstore/naval"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "phi3"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
CONF_THRESH = 0.25

# ---------- ALWAYS RENDER UI ----------
st.write("Welcome aboard, Commander 🚢")

# ---------- OLLAMA CHECK ----------
def check_ollama():
    try:
        subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        st.sidebar.warning("⚠️ Ollama not running — start it via `ollama serve`.")
        return False

ollama_ok = check_ollama()

# ---------- EMBEDDINGS ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    if ollama_ok:
        try:
            subprocess.run(["ollama", "show", EMBED_MODEL], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            emb = OllamaEmbeddings(model=EMBED_MODEL)
            emb.embed_query("test")
            st.sidebar.success(f"✅ Using Ollama embeddings ({EMBED_MODEL})")
            return emb
        except Exception as e:
            st.sidebar.warning(f"Ollama embeddings failed: {e}")
    st.sidebar.info("Using HuggingFace MiniLM instead")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# ---------- LOAD PDFS ----------
def find_pdfs(root):
    pdfs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, f))
    return pdfs

def load_pdf(path):
    try:
        docs = PyPDFLoader(path).load()
    except:
        docs = UnstructuredPDFLoader(path, strategy="ocr_only").load()
    for d in docs:
        d.metadata["source"] = path
    return docs

# ---------- BUILD VECTORSTORE ----------
@st.cache_resource(show_spinner=True)
def build_vectorstore(force=False):
    pdfs = find_pdfs(DATA_DIR)
    if not pdfs:
        st.warning("📂 No PDFs found in data/ folder — add some first.")
        return None
    all_docs = []
    progress = st.progress(0)
    for i, pdf in enumerate(pdfs, 1):
        all_docs.extend(load_pdf(pdf))
        progress.progress(i / len(pdfs))
    if not all_docs:
        st.error("⚠️ No readable text found in PDFs.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return db

rebuild = st.sidebar.checkbox("🔄 Rebuild Index", value=False)
db = build_vectorstore(force=rebuild)

# ---------- MODEL ----------
try:
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    st.sidebar.success(f"🧠 Model loaded: {CHAT_MODEL}")
except Exception as e:
    st.sidebar.warning(f"Chat model not loaded ({e})")
    llm = None

# ---------- MAIN Q&A ----------
st.markdown("---")
q = st.text_input("Ask about ships or submarines:")

if q:
    if not db:
        st.error("❌ No database found. Add PDFs and click 'Rebuild Index'.")
    elif not llm:
        st.error("❌ LLM model not running. Start Ollama or check model name.")
    else:
        with st.spinner("🔎 Searching naval database..."):
            results = db.similarity_search_with_score(q, k=RETRIEVAL_K)
            filtered = [doc.page_content for doc, score in results if score <= CONF_THRESH]
            if not filtered and results:
                filtered = [results[0][0].page_content]
            if not filtered:
                st.warning("No relevant information found.")
                st.success("No information about this.")
            else:
                context = "\n\n".join(filtered)
                prompt = f"""
You are a Naval Data AI.
Answer ONLY from the context below.
If no relevant info exists, say exactly: 'No information about this.'

Context:
{context}

Question: {q}
Answer:
"""
                resp = llm.invoke(prompt)
                ans = getattr(resp, "content", str(resp)).strip()
                answer = ans if ans else "No information about this."
                st.success(answer)
                with st.expander("📘 Source Texts"):
                    for txt in filtered:
                        st.write(txt[:600] + ("…" if len(txt) > 600 else ""))