# 🤖 LLM-Powered Document Agent with Qdrant & LangChain

Welcome to **LLM-Powered Document Agent**, a modular Python application that combines the power of **LangChain**, **Qdrant**, **OpenAI**, and **custom tools** to create an intelligent assistant capable of understanding and answering questions based on your documents.

---

## 🚀 Features

- ✅ **PDF Chunking** – Extracts and splits text from PDFs into manageable chunks using token limits.
- ✅ **Embedding Generation** – Converts text chunks into vector embeddings using OpenAI.
- ✅ **Vector Storage** – Stores and retrieves vectors efficiently using Qdrant as a vector database.
- ✅ **Custom Tools** – Register your own tools (e.g., PDF parsing, logging, chunking) via `StructuredTool`.
- ✅ **Agent Execution** – Use LangChain to build a Tool-Using LLM Agent that responds with reasoning.

---

## 🧠 Tech Stack

| Component       | Description                                  |
|----------------|----------------------------------------------|
| **LangChain**   | Framework for building LLM apps              |
| **Qdrant**      | Vector database for semantic search          |
| **OpenAI**      | Embeddings & LLMs (like GPT-4 or GPT-3.5)    |
| **PyPDF2**      | Extracts text from PDF files                 |
| **Tiktoken**    | Tokenizer compatible with OpenAI embeddings  |
| **Pydantic**    | Used to validate and structure tool inputs   |

---

## 📁 Project Structure

```
project_root/
│
├── app/
│   ├── __init__.py
│   ├── tools/                      # Custom tools (e.g., pdf_to_chunks, add_documents)
│   │   ├── pdf_tool.py
│   │   ├── qdrant_tool.py
│   │   └── schema.py              # Pydantic input schemas for tools
│   ├── agent.py                   # LLM Agent creation and execution
│   ├── logger.py                  # Logger setup using Python logging
│   └── config.py                  # API keys, environment setup
│
├── tests/                         # Unit tests
│
├── data/                          # Example PDFs
├── .env                           # API keys and secrets
├── pyproject.toml                 # Poetry configuration
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llm-doc-agent.git
cd llm-doc-agent
```

### 2. Install Dependencies (using Poetry)

```bash
poetry install
poetry shell
```

### 3. Configure Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
QDRANT_HOST=http://localhost:6333
```

Make sure Qdrant is running locally, e.g.:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 🛠 How It Works

### ➤ Step 1: Chunk a PDF

```python
chunks = pdf_to_chunks("data/sample.pdf")
```

### ➤ Step 2: Add Chunks to Qdrant

```python
add_documents_to_collection("my_collection", documents=chunks)
```

### ➤ Step 3: Run the Agent

```python
response = agent.invoke("What is this document about?")
print(response)
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 💡 Future Ideas

- 🔍 Add multi-turn memory to the agent
- 🌐 Deploy as a FastAPI or Streamlit app
- 🧩 Integrate LangGraph for advanced workflows
- 🗃 Add support for other file formats (e.g., .docx, .html)

---

## 🙌 Credits

- Developed by **Pablo Marín** 🇨🇴  
- Based on tools from [LangChain](https://www.langchain.com/), [Qdrant](https://qdrant.tech/), and [OpenAI](https://openai.com/)

