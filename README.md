

# Research Paper Assistant using RAG

## Overview

This project is a **research paper assistant** that leverages **Retrieval-Augmented Generation (RAG)** to help users analyze research papers. The system allows uploading one or multiple PDFs, extracts and chunks the text, creates embeddings for semantic retrieval, and uses the **Groq LLM** to answer user queries based on the content of the papers.

The project uses **LangChain** for the RAG pipeline, **HuggingFace embeddings** for vectorization, and **Streamlit** for a simple web-based interface.

---
![WhatsApp Image 2025-10-11 at 16 17 20_d9f8a3a8](https://github.com/user-attachments/assets/5c1e1bb2-605f-4342-8fa3-6a027a876870)
![WhatsApp Image 2025-10-11 at 16 17 45_99c3381d](https://github.com/user-attachments/assets/0a43f24a-a239-4432-8bd2-1ccf11b256c5)
![WhatsApp Image 2025-10-11 at 16 20 00_b3f7fce9](https://github.com/user-attachments/assets/c33e5be4-d301-464d-b2e2-7f272e1c286a)

## Features

* Upload one or more PDF research papers.
* Extract text from PDFs and split it into semantic chunks.
* Create vector embeddings for each chunk to enable fast retrieval.
* Ask natural language questions about the paper(s) and get context-aware answers.
* Retrieve and display the most relevant source chunks alongside answers.
* Supports multiple PDFs in a single session.
* Uses Groq LLM for answer generation and HuggingFace embeddings for vectorization.

---

## Tech Stack

* **Python 3.10+**
* **Streamlit** – Web-based UI for file uploads and interactive querying.
* **LangChain** – For building the RAG pipeline.
* **ChromaDB** – Vector database for storing embeddings.
* **HuggingFace Sentence-Transformers** – To generate embeddings locally.
* **Groq LLM(LLaMA 3.1 8B Instant)** – Language model used for generating responses.
* **PyMuPDF (fitz)** – For extracting text from PDF files.
* **dotenv** – For securely storing API keys.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Nourin04/Research-Paper-Assistant-using-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```



### 4. Setup API Keys

* Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

* Replace `your_groq_api_key_here` with your actual Groq API key.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage Instructions

1. **Upload PDFs** – Click the file uploader and select one or more research papers in PDF format.
2. **Text Extraction & Chunking** – The system will automatically extract text and split it into smaller chunks.
3. **Vector Embeddings** – Chunks are converted into embeddings for semantic search.
4. **Ask Questions** – Enter a natural language question in the input box to query the papers.
5. **View Answers** – The Groq LLM will provide answers based on the most relevant chunks.
6. **View Source Contexts** – Expand the context panel to see which chunks were used for answering.

---

## Project Structure

```
rag_research_assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (Groq API)
└── README.md               # Project documentation
```

---

## Notes

* **Groq API** is only used for generating answers (LLM). Embeddings are generated locally using HuggingFace.
* **Vectorstore persistence** uses temporary directories. Restarting the app clears existing embeddings.
* The retriever fetches the top `k` chunks (default `k=6`) for context. Adjust `k` for better accuracy.
* For larger papers, embedding generation may take some time depending on system resources.

---

## Future Improvements

* Save vectorstore to disk for permanent storage across sessions.
* Add summarization for uploaded PDFs automatically.
* Support more advanced prompts for better answer quality.
* Improve UI with sections for each PDF and page numbers.

---
