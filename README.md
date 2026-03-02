# Smart Contract Analysis System

A comprehensive document analysis system to extract insights, answer questions, and analyze smart contracts and other complex documents using modern retrieval and generation techniques.

## Why We Use These Technologies

This project uses a combination of modern technologies to ensure accurate, context-aware answers grounded in your actual documents. Here is why we use each part of the stack:

### 1. The API (LLM Generation)
We use a Large Language Model API (configured via Groq) as the core reasoning engine. 
**Why we use it:** The LLM receives the relevant contract clauses and the user's question, applying logic and natural language understanding to generate a clear, concise, and accurate answer. 
**Comparison vs Alternatives:** We use Groq because it currently offers a generous free API tier with incredibly fast inference speeds for open-source models (like Llama). This makes it vastly superior for cost-conscious, high-performance projects compared to paid, proprietary APIs like OpenAI (ChatGPT) or Anthropic (Claude), which charge per token and can quickly become expensive during development and testing.

### 2. Vector Database (FAISS)
We use FAISS (Facebook AI Similarity Search) as our local vector database.
**Why we use it:** Instead of reading through hundreds of pages of contracts every time a question is asked, the vector database stores mathematical representations (vectors) of every paragraph in your documents for instant retrieval. 
**Comparison vs Alternatives:** We chose FAISS because it is exceptionally lightweight, runs entirely in memory, and saves directly to local disk files. It is much easier to set up than heavy client-server vector databases like Milvus, Pinecone, or Qdrant which require Docker containers or cloud accounts. Furthermore, it avoids the complex file-locking and SQLite dependency issues that frequently plague other local databases like ChromaDB on Windows and various deployment environments.

### 3. Embeddings (HuggingFace Sentence-Transformers)
We use a local, free, open-source embedding model (`all-MiniLM-L6-v2`) to convert text into numbers.
**Why we use it:** The embedding model translates text into high-dimensional vectors that capture the *meaning* of the text, allowing the system to match concepts rather than just exact keywords.
**Comparison vs Alternatives:** We use local HuggingFace embeddings because they are 100% free and private. Your document text never leaves your machine to be embedded. In contrast, using OpenAI's `text-embedding-3` or similar APIs would incur monetary costs for every document uploaded and transmit your sensitive text to a third-party server, which is often a dealbreaker for legal or confidential contracts.

### 4. Frontend Web UI (Gradio)
We use Gradio for the user interface.
**Why we use it:** Gradio provides an intuitive platform where users can easily upload their documents in one tab, and seamlessly chat with the system in another tab.
**Comparison vs Alternatives:** Gradio allows us to build a full, interactive web interface directly in Python, entirely avoiding the steep learning curves and heavy Node.js dependencies of frontend frameworks like React, Vue, or Angular. Compared to Streamlit (another Python framework), Gradio provides a superior, out-of-the-box Chatbot component that maintains state efficiently and is specifically optimized for natural language applications.

### 5. Backend Server (FastAPI)
We use FastAPI to serve our REST API (used when running `python main.py serve`).
**Why we use it:** FastAPI allows us to instantly create robust, production-ready API endpoints that external applications or microservices can query.
**Comparison vs Alternatives:** We use FastAPI instead of Flask or Django because FastAPI is natively asynchronous (which is essential for handling slow LLM network requests efficiently without blocking) and automatically generates OpenAPI (Swagger) documentation. Flask requires heavy manual configuration for async operations and API docs, while Django is far too monolithic and heavy for a lightweight inference microservice.

---

## How to Use the Project

### Prerequisites
Make sure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
1. Copy the `.env.example` file to create a `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and add your `GROQ_API_KEY`. (You can get a free API key from the Groq console).

### Running the Application

There are multiple ways to interact with the project:

#### 1. Launch the User Interface (Recommended)
To launch the Gradio web interface, run:
```bash
python main.py ui
```
- A local URL (usually `http://127.0.0.1:7860`) will appear in the terminal.
- Open this link in your browser.
- **Step 1:** Go to the "Ingestion" tab and upload your contract documents (PDF, DOCX, TXT). Click "Ingest & Save Index".
- **Step 2:** Go to the "Chat" tab and ask questions about your documents!

#### 2. Running an API Server
If you want to integrate this system into another application, you can start the REST API:
```bash
python main.py serve
```
This will start a FastAPI server at `http://localhost:8000` where you can send HTTP requests to analyze documents.

#### 3. Command Line Ingestion
If you prefer to ingest documents via the terminal before launching the UI:
```bash
python main.py ingest --files my_contract.pdf another_contract.docx
```

---

## Features
- **Accurate Document Retrieval**: Only answers questions based on the uploaded contracts.
- **Source Citations**: Tells you exactly which file and page the answer came from.
- **Evaluation Pipeline**: Includes an integrated evaluation script (`python main.py evaluate`) to test answer accuracy against known keywords.
- **Modular Design**: Separated into distinct logic blocks for chunking, embedding, vector storage, and generation.
