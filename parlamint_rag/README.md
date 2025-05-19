# ParlaMint RAG Project

This project implements a Retrieval Augmented Generation (RAG) system specifically tailored for querying and interacting with the ParlaMint corpus. The system processes TEI XML data from ParlaMint, chunks it, generates embeddings, builds a FAISS index for efficient similarity search, and provides a multi-stage retrieval mechanism (FAISS + cross-encoder re-ranking) to find the most relevant text segments for a given query.

## Project Structure

```
parlamint_rag/
├── data/                   # Holds the ParlaMint corpus data
│   ├── raw/                # Stores original ParlaMint TEI XML files (e.g., dummy_parlamint_en_01.xml)
│   └── processed/          # Stores processed data, like chunked text (e.g., parlamint_chunks.jsonl)
├── index/                  # Contains generated FAISS indexes and associated metadata
│   ├── embeddings.npy      # Saved NumPy array of text chunk embeddings
│   ├── embedding_metadata.pkl # Metadata corresponding to each embedding
│   ├── parlamint_faiss.index # The saved FAISS index file
│   └── parlamint_metadata.pkl# Metadata list mapped to FAISS index IDs
├── notebooks/              # Jupyter notebooks for experimentation and analysis (if any)
│   └── .gitkeep            # Placeholder
├── src/                    # Source code for the RAG system
│   ├── __init__.py         # Makes src a Python package
│   ├── config.py           # Configuration hub (paths, model names, parameters) using Pydantic
│   ├── data_loader.py      # Loads and parses ParlaMint TEI XML files
│   ├── text_processor.py   # Chunks text data into manageable pieces
│   ├── embedding_generator.py # Generates vector embeddings for text chunks
│   ├── faiss_indexer.py    # Creates, saves, and loads FAISS indexes
│   ├── retriever.py        # Implements the multi-stage retrieval (FAISS + re-ranking) logic
│   └── main.py             # Main CLI entry point to build the index and query the system
├── requirements.txt        # Lists Python package dependencies for the project
└── README.md               # This file: project overview, setup, and usage instructions
```

## RAG Pipeline Overview

The system follows a structured pipeline to enable efficient and relevant information retrieval:

1.  **Data Loading and Parsing (`src/data_loader.py`):**
    *   **Input:** ParlaMint TEI XML files located in `data/raw/`.
    *   **Process:** Reads XML files, parses them using `lxml`, and extracts relevant textual content (e.g., utterances, segments) along with metadata (speaker, document ID, etc.).
    *   **Output:** A list of dictionaries, where each dictionary represents an utterance or a significant text block.

2.  **Text Chunking (`src/text_processor.py`):**
    *   **Input:** The list of text blocks from the data loader.
    *   **Process:** Divides the text from each block into smaller, overlapping chunks. Chunk size and overlap are defined in `src/config.py`. This ensures that semantic context is preserved and that information isn't lost at chunk boundaries.
    *   **Output:** A list of chunk dictionaries, each containing the chunk text and its associated metadata. These are typically saved to a JSONL file in `data/processed/` (e.g., `parlamint_chunks.jsonl`).

3.  **Embedding Generation (`src/embedding_generator.py`):**
    *   **Input:** The list of text chunks.
    *   **Process:** Uses a pre-trained sentence-transformer model (defined in `src/config.py`, e.g., `sentence-transformers/all-mpnet-base-v2`) to convert each text chunk into a dense vector representation (embedding). These embeddings capture the semantic meaning of the text.
    *   **Output:**
        *   A NumPy array of embeddings (`embeddings.npy`).
        *   A corresponding list of metadata for each embedding (`embedding_metadata.pkl`), saved in the `index/` directory.

4.  **FAISS Indexing (`src/faiss_indexer.py`):**
    *   **Input:** The generated embeddings (NumPy array) and their metadata.
    *   **Process:** Builds a FAISS (Facebook AI Similarity Search) index from the embeddings. FAISS allows for highly efficient similarity searches even with millions of vectors. The index type (e.g., `IndexFlatL2`, `HNSWFlat`) is configurable in `src/config.py`.
    *   **Output:**
        *   The FAISS index file (e.g., `parlamint_faiss.index`).
        *   A metadata file (`parlamint_metadata.pkl`) that maps FAISS index IDs back to the original chunk metadata, saved in the `index/` directory.

5.  **Multi-Stage Retrieval (`src/retriever.py` and `src/main.py` for querying):**
    *   **Input:** A user query (string).
    *   **Process:**
        1.  **Query Embedding:** The user's query is converted into an embedding using the same sentence-transformer model used for chunk embedding.
        2.  **Initial Retrieval (FAISS Search):** The query embedding is used to search the FAISS index, retrieving the top-k most similar chunk embeddings (and thus chunks). This step is very fast.
        3.  **Re-ranking (Cross-Encoder):** The top-k candidate chunks are then re-evaluated using a more powerful (but slower) CrossEncoder model (defined in `src/config.py`, e.g., `cross-encoder/ms-marco-T5-Base`). The CrossEncoder directly compares the query with each candidate chunk text to provide a more accurate relevance score.
    *   **Output:** A ranked list of the top-n most relevant text chunks, along with their content and metadata, displayed to the user via the CLI.

## Setup and Usage

(This section can be expanded with detailed setup and command examples later)

1.  **Prerequisites:**
    *   Python 3.9+
    *   Ollama (if planning to integrate an Ollama LLM for generation later)

2.  **Installation:**
    ```bash
    git clone <repository-url>
    cd parlamint_rag
    pip install -r requirements.txt
    ```

3.  **Data Preparation:**
    *   Place your ParlaMint TEI XML files into the `data/raw/` directory.
    *   For initial testing, `python -m parlamint_rag.src.data_loader` will generate dummy English XML files in `data/raw/`.

4.  **Building the Index:**
    Run the main script with the `build` action. Use `--max_files` for testing with a subset of your data and `--force_reprocess` to rebuild all artifacts.
    ```bash
    # Build index using a maximum of 2 files from data/raw, forcing all steps
    python -m parlamint_rag.src.main build --max_files 2 --force_reprocess

    # Build index using all files in data/raw
    python -m parlamint_rag.src.main build
    ```

5.  **Querying the System:**
    Once the index is built, use the `query` action:
    ```bash
    python -m parlamint_rag.src.main query -q "your search query here"
    ```
    Example with dummy data:
    ```bash
    python -m parlamint_rag.src.main query -q "John Doe speaking about tests"
    ```

This provides a foundational RAG system. Future enhancements could include integrating a Large Language Model (LLM) like those available via Ollama to generate natural language responses based on the retrieved chunks. 