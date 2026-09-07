import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stTextInput input { 
        border: 2px solid #cbd5e1; 
        border-radius: 8px; 
        padding: 12px; 
        font-size: 16px;
    }
    .stTextInput input:focus { border-color: #1e3a8a; }
    .stButton button { 
        background-color: #1e3a8a; 
        color: white; 
        border-radius: 8px; 
        font-weight: bold; 
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton button:hover { background-color: #152c6b; border-color: #152c6b; color: white;}
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# --- PDF Processing Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_rag_chain():
    prompt_template = """
    Answer the question as detailed as possible based ONLY on the provided context. 
    If the answer is not contained in the context, explicitly state: "The answer is not available in the uploaded document." 
    Do not guess or make up information.

    Context:
    {context}
    
    Question: 
    {question}

    Answer:
    """
    
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("⚠️ Missing GROQ_API_KEY. Please set it in Streamlit Cloud Secrets or your .env file.")
        st.stop()

    model = ChatGroq(
        api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    
    # Modern LCEL chain
    return prompt | model | output_parser

# --- Main App ---
def main():
    st.title("📚 Chat with your Large PDFs")
    st.write("Upload your documents and query them using Groq-accelerated inference.")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.header("📂 Document Pipeline")
        st.write("1. Upload files\n2. Click Process\n3. Start chatting!")
        
        pdf_docs = st.file_uploader("Upload PDF Files:", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF.")
            else:
                with st.spinner("Extracting text and building vector store..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        st.success("✅ Documents processed successfully!")
                    else:
                        st.error("❌ Could not extract text. Check if the PDF consists of scanned images.")

    st.markdown("---")
    
    user_question = st.text_input("Ask a question about your uploaded documents:", placeholder="e.g., What is the summary of this document?")

    if user_question:
        if st.session_state.vector_store is None:
            st.error("⚠️ Please upload and process a PDF in the sidebar first.")
        else:
            with st.spinner("Retrieving context and generating response..."):
                # Retrieve matching chunks
                docs = st.session_state.vector_store.similarity_search(user_question, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                chain = get_rag_chain()
                response = chain.invoke({"context": context, "question": user_question})
                
                st.markdown("### 💡 AI Response:")
                st.info(response)

if __name__ == "__main__":
    main()
