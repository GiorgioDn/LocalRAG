# LocalRAG

A local-first Retrieval-Augmented Generation (RAG) application that lets you ingest your own documents and ask questions grounded in their content. Everything runs on your machine: embeddings and chat go through [Ollama](https://ollama.com/), and vectors are stored locally with [ChromaDB](https://www.trychroma.com/).

## Not an official product, this app was created solely for learning and personal use.

## Development


| Phase          | Model       |
| -------------- | ----------- |
| Planning       | GPT 5.4     |
| Implementation | Gemma 4:12b |


## Description

LocalRAG is a Python CLI tool for building a private knowledge base from text and PDF files. It follows a classic RAG pipeline:

1. **Ingestion** — Load a document, normalize its text, and split it into overlapping chunks.
2. **Indexing** — Embed each chunk with an Ollama embedding model and persist the vectors in a local Chroma collection.
3. **Retrieval** — When you ask a question, embed the query and fetch the most relevant chunks.
4. **Generation** — Build a grounded prompt from the retrieved context and send it to an Ollama chat model.

The project is organized into focused modules:

- **Ingestion** — Text loader (`.txt`, `.md`, `.csv`, `.json`) and PDF loader (PyMuPDF), plus text normalization and chunking.
- **Vector store** — Chroma connector for collection management and similarity search.
- **Models** — Ollama clients for embeddings and chat.
- **RAG** — Indexer, retriever, prompt builder, generator, and shared pipeline functions.
- **CLI entry points** — `main_text.py` for text files and `main_pdf.py` for PDFs.

All configuration is driven by environment variables, so you can swap models and tune chunking without changing code.

## Use of the program



### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- Ollama models pulled for both chat and embeddings, for example:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```



### Installation

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd LocalRAG
pip install -r requirements-dev.txt
```



### Configuration

Copy the example environment file and adjust the values for your setup:

```bash
cp .env.example .env
```

Load the variables into your shell (or use a tool such as `direnv`):

```bash
set -a && source .env && set +a
```


| Variable             | Description                                   |
| -------------------- | --------------------------------------------- |
| `OLLAMA_BASE_URL`    | Base URL of the local Ollama server           |
| `OLLAMA_CHAT_MODEL`  | Model used to generate answers                |
| `OLLAMA_EMBED_MODEL` | Model used to embed documents and queries     |
| `CHROMA_PERSIST_DIR` | Directory where Chroma persists vector data   |
| `DEFAULT_COLLECTION` | Default collection name for ingestion and Q&A |
| `CHUNK_SIZE`         | Maximum characters per chunk                  |
| `CHUNK_OVERLAP`      | Overlap between consecutive chunks            |
| `TOP_K`              | Number of chunks retrieved per question       |




### Text-based RAG

Ingest a text file and optionally ask a question in the same run:

```bash
# Ingest only
python -m src.rag_app.main_text path/to/document.txt

# Ingest and ask a question
python -m src.rag_app.main_text path/to/document.txt \
  --collection my_docs \
  --question "What is the main topic of this document?"
```

Supported text extensions: `.txt`, `.md`, `.csv`, `.json`.

### PDF-based RAG

Ingest a PDF and optionally ask a question:

```bash
# Ingest only
python -m src.rag_app.main_pdf path/to/document.pdf

# Ingest and ask a question
python -m src.rag_app.main_pdf path/to/document.pdf \
  --collection my_pdfs \
  --question "Summarize the key findings."
```

Page-level metadata is preserved during ingestion, so answers can reference the source file and page number.

### Output

When you pass `--question`, the CLI prints:

- **Answer** — The model's response, grounded in the retrieved context.
- **Sources** — The documents (and pages, for PDFs) used to build the answer.

If you omit `--question`, the program only indexes the file into the vector store. You can run it again later with a question against the same collection.

### Running tests

```bash
pip install -r requirements-dev.txt
pytest
```



### Programmatic usage

You can also call the pipeline directly from Python:

```python
from src.rag_app.rag.pipeline import ingest_text_path, answer_question

ingest_text_path("notes.md", collection_name="notes")
result = answer_question("What are the action items?", collection_name="notes")

print(result.answer)
for source in result.sources:
    print(source)
```

