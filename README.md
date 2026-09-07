# 📚 StandardRAGapp: Intelligent PDF Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url-here.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-white)](#)
[![Groq](https://img.shields.io/badge/⚡-Groq_LPU-orange)](#)

StandardRAGapp is a production-ready Retrieval-Augmented Generation (RAG) application. It allows users to upload large, multi-page PDF documents and chat with their data instantly. Powered by **Groq's LPU inference engine**, **LangChain Expression Language (LCEL)**, and **FAISS** vector search, this app delivers lightning-fast semantic search and accurate responses.

## ✨ Features
* **Document Processing:** Efficient character-level text splitting to handle massive PDFs.
* **Open-Source Embeddings:** Utilizes HuggingFace's `all-MiniLM-L6-v2` for high-quality, local vectorization.
* **In-Memory Vector DB:** Seamless semantic search powered by FAISS.
* **Ultra-Fast Generation:** Integrates `openai/gpt-oss-120b` running on Groq for near-instant answers.
* **Interactive UI:** A clean, responsive interface built entirely in Python using Streamlit.

## 🏗️ System Architecture
1. **Document Ingestion:** PyPDF2 extracts text from uploaded PDFs.
2. **Chunking & Embedding:** Recursive splitting breaks text into 1000-character chunks, which are embedded via HuggingFace models.
3. **Storage:** Vectors are indexed in a local FAISS database.
4. **Retrieval & Generation:** User queries fetch the top 4 most relevant chunks, passing them into an LCEL chain alongside a strict context-bound prompt.

## 🚀 Run it Locally

### Prerequisites
* Python 3.9+
* A [Groq API Key](https://console.groq.com)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/M-Muneeb87/StandardRAGapp.git](https://github.com/M-Muneeb87/StandardRAGapp.git)
   cd StandardRAGapp
