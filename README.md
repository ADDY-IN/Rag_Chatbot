# ⚓ Naval Intelligence RAG Chatbot — Prototype

An AI-powered Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **Ollama**, and **Streamlit**.  
This prototype processes large-scale **naval documentation** (ships, submarines, and technical PDFs) to provide precise, context-based answers — directly extracted from your uploaded or indexed files.

---

## 🚀 Features

- ⚙️ **RAG Pipeline** — Combines document embeddings + vector retrieval + LLM inference.  
- 📚 **Multi-Folder Document Indexing** — Scans and indexes all PDFs inside the `data/` folder (including nested directories).  
- 🧩 **Embeddings via Ollama** — Uses `nomic-embed-text` model for local embeddings (falls back to HuggingFace if unavailable).  
- 🧠 **Local LLM Integration** — Works with `phi3`, `llava`, or any other Ollama-supported model.  
- 🖼️ **Image Support Ready** — Extendable for multimodal (text + image) documents.  
- 🌐 **Streamlit UI** — Interactive and lightweight prototype dashboard.

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **LangChain** | Framework for chaining LLM + retriever logic |
| **Ollama** | Local language model and embedding host |
| **Streamlit** | Web UI framework |
| **ChromaDB** | Vector store for efficient retrieval |
| **HuggingFace** | Backup embedding generator (MiniLM) |

---

## 📁 Project Structure

Rag_Chatbot_Project/
│
├── main.py                 # Streamlit app and RAG logic
├── requirements.txt        # Dependencies list
├── data/                   # Naval PDFs and documents
├── vectorstore/            # Chroma vector database
├── .env                    # Optional environment file
└── README.md               # Project overview

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Rag_Chatbot_Project.git
cd Rag_Chatbot_Project

2️⃣ Create Virtual Environment

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Requirements

pip install -r requirements.txt

4️⃣ Start Ollama

Make sure Ollama is installed and running:

ollama serve
ollama pull phi3
ollama pull nomic-embed-text

5️⃣ Run the App

streamlit run main.py

⸻

🔒 Notes
	•	This prototype is designed for local testing and client demo.
	•	For production use:
	•	Host embeddings and models on a secure cloud backend.
	•	Implement caching, async queries, and improved retrievers.
	•	Add authentication and logging for enterprise use.

⸻

🧑‍💻 Author

Aditya Kaushik
📍 Built for client demo — Naval RAG Chatbot (2025)
💬 Powered by Ollama, LangChain, and Streamlit
