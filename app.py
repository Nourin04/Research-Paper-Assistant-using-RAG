import os
from dotenv import load_dotenv
import streamlit as st
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit page config
st.set_page_config(page_title="RAG-based Research Paper Assistant", layout="wide")
st.title(" Research Paper Assistant using RAG")
st.write("Upload one or more research paper PDFs to begin analyzing them.")

# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# --- PDF Upload ---
uploaded_files = st.file_uploader(
    "📎 Upload research paper PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.pdf_names:
            st.session_state.pdf_names.append(uploaded_file.name)

            # Extract text from PDF
            text = ""
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for i, page in enumerate(pdf_document):
                page_text = page.get_text()
                text += page_text

            st.write(f"### Extracted Text Preview for {uploaded_file.name}:")
            st.text_area("Text Preview", text[:2000], height=200)

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_text(text)
            st.write(f"✅ {uploaded_file.name} split into {len(chunks)} chunks.")

            # Store chunks with metadata (PDF name & page info)
            metadata_chunks = [{"source": uploaded_file.name} for _ in chunks]
            st.session_state.all_chunks.extend(list(zip(chunks, metadata_chunks)))

    # --- Embeddings + Vector Store ---
    if st.session_state.all_chunks:
        with st.spinner("Creating embeddings and building vector store..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            persist_directory = tempfile.mkdtemp()

            # Separate texts and metadata
            texts, metadatas = zip(*st.session_state.all_chunks)

            st.session_state.vectorstore = Chroma.from_texts(
                texts,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=persist_directory
            )
        st.success("✅ Embeddings created and stored in vector database!")

# --- RAG QA Section ---
if st.session_state.vectorstore is not None:
    st.subheader(" Ask Questions About Your Research Papers")

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    # Create retriever
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})

    # Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # User input
    user_question = st.text_input("🔍 Ask a question about the paper(s):")

    if user_question:
        with st.spinner("Generating answer using Groq LLM..."):
            result = qa_chain({"query": user_question})

            # Display answer
            st.write("###  Answer:")
            st.write(result["result"])

            # Display retrieved context snippets with source info
            with st.expander(" View Retrieved Contexts"):
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Source:** {source}")
                    st.markdown(doc.page_content[:400] + "...")
