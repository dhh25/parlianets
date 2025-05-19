#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Command-Line Interface (CLI) for the ParlaMint RAG System.

This script serves as the main entry point for interacting with the RAG system.
It provides two primary functionalities:
1. `build`: Processes raw ParlaMint TEI data, creates text chunks, generates embeddings,
   and builds a FAISS index. This prepares all necessary data artifacts for querying.
2. `query`: Takes a user query, performs a multi-stage retrieval (FAISS search + re-ranking),
   and displays the most relevant text chunks using the `rich` library for formatted output.

Key Functions:
- `ensure_data_pipeline()`: Manages the data preparation steps (loading, chunking, embedding, indexing),
  allowing for reprocessing if needed and utilizing existing artifacts to save time.
- `perform_query()`: Handles the user query, loads necessary models and indexes, invokes the
  retrieval process, and formats the results for display.
- `main()`: Parses command-line arguments (`build` or `query`, query text, processing options)
  and dispatches to the appropriate functions.

Uses modules:
- `config`: For all system configurations.
- `data_loader`: To load and parse raw TEI XML data.
- `text_processor`: To chunk loaded text data.
- `embedding_generator`: To create vector embeddings for chunks.
- `faiss_indexer`: To build and manage the FAISS similarity search index.
- `retriever`: To perform the actual search and re-ranking against the query.

The CLI uses `argparse` for argument parsing and `rich` for enhanced terminal output
(progress bars, tables, formatted text).
"""

import argparse
from pathlib import Path
import time
import sys
from typing import Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from .config import config
from .data_loader import load_parlamint_data
from .text_processor import create_chunks, save_chunks_to_jsonl, load_chunks_from_jsonl
from .embedding_generator import (get_embedding_model as get_sbert_model, 
                                  generate_embeddings, 
                                  save_embeddings_and_metadata, 
                                  load_embeddings_and_metadata)
from .faiss_indexer import (create_faiss_index, 
                            save_faiss_index, 
                            load_faiss_index)
from .retriever import get_reranker_model, retrieve_and_rerank

console = Console()

def ensure_data_pipeline(max_files_raw_data: int = 0, force_reprocess: bool = False) -> Tuple[Path, Path, Path]:
    """Ensures that data is loaded, chunked, embedded, and indexed."""
    
    processed_chunks_file = Path(config.paths.processed_data_dir) / "parlamint_chunks.jsonl"
    embeddings_file = Path(config.paths.index_dir) / "embeddings.npy"
    embedding_metadata_file = Path(config.paths.index_dir) / "embedding_metadata.pkl"
    faiss_index_file = Path(config.paths.index_dir) / config.faiss.index_filename
    faiss_metadata_file = Path(config.paths.index_dir) / config.faiss.metadata_filename

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=False, # Keep progress displayed after completion
    ) as progress:

        # --- 1. Load and Chunk Data --- 
        task_chunk = progress.add_task("Processing raw data...", total=1)
        if not processed_chunks_file.exists() or force_reprocess:
            progress.update(task_chunk, description="Loading raw ParlaMint TEI files...")
            raw_docs = load_parlamint_data(max_files=max_files_raw_data)
            if not raw_docs:
                console.print("[bold red]No raw documents loaded. Exiting.[/bold red]")
                sys.exit(1)
            
            progress.update(task_chunk, description="Creating text chunks...")
            chunks = create_chunks(raw_docs, config.processing.chunk_size, config.processing.chunk_overlap)
            if not chunks:
                console.print("[bold red]No chunks created. Exiting.[/bold red]")
                sys.exit(1)

            progress.update(task_chunk, description=f"Saving {len(chunks)} chunks to {processed_chunks_file}...")
            save_chunks_to_jsonl(chunks, processed_chunks_file)
            console.print(f":floppy_disk: Chunks saved to [cyan]{processed_chunks_file}[/cyan]")
        else:
            console.print(f":file_folder: Using existing processed chunks from [cyan]{processed_chunks_file}[/cyan]")
        progress.update(task_chunk, completed=1, description="Raw data processed.")

        # --- 2. Generate Embeddings --- 
        task_embed = progress.add_task("Generating embeddings...", total=1)
        if not embeddings_file.exists() or not embedding_metadata_file.exists() or force_reprocess:
            console.print(f"Loading chunks from {processed_chunks_file} for embedding...")
            # Load all chunks into memory for embedding. For very large datasets, consider batching.
            all_chunks_for_embedding = list(load_chunks_from_jsonl(processed_chunks_file))
            if not all_chunks_for_embedding:
                console.print(f"[bold red]No chunks loaded from {processed_chunks_file}. Cannot generate embeddings. Exiting.[/bold red]")
                sys.exit(1)

            progress.update(task_embed, description=f"Generating embeddings for {len(all_chunks_for_embedding)} chunks...")
            sbert_model = get_sbert_model() # Uses model from config
            embeddings_np = generate_embeddings(all_chunks_for_embedding)
            
            progress.update(task_embed, description=f"Saving embeddings and metadata...")
            save_embeddings_and_metadata(embeddings_np, all_chunks_for_embedding, Path(config.paths.index_dir))
            console.print(f":brain: Embeddings and their metadata saved to [cyan]{config.paths.index_dir}[/cyan]")
        else:
            console.print(f":file_folder: Using existing embeddings from [cyan]{embeddings_file}[/cyan] and metadata from [cyan]{embedding_metadata_file}[/cyan]")
        progress.update(task_embed, completed=1, description="Embeddings generated.")

        # --- 3. Create FAISS Index --- 
        task_faiss = progress.add_task("Creating FAISS index...", total=1)
        if not faiss_index_file.exists() or not faiss_metadata_file.exists() or force_reprocess:
            progress.update(task_faiss, description=f"Loading embeddings for FAISS index...")
            # The metadata saved by save_embeddings_and_metadata is the list of chunk dicts
            embeddings_np_for_faiss, chunk_metadata_for_faiss = load_embeddings_and_metadata(Path(config.paths.index_dir))
            if embeddings_np_for_faiss.size == 0 or not chunk_metadata_for_faiss:
                console.print("[bold red]Failed to load embeddings or metadata for FAISS indexing. Exiting.[/bold red]")
                sys.exit(1)

            progress.update(task_faiss, description=f"Creating FAISS index ({config.faiss.index_type})...")
            idx = create_faiss_index(embeddings_np_for_faiss, config.faiss.index_type)
            
            progress.update(task_faiss, description=f"Saving FAISS index and metadata map...")
            save_faiss_index(idx, chunk_metadata_for_faiss, Path(config.paths.index_dir))
            console.print(f":mag: FAISS index and its metadata map saved to [cyan]{config.paths.index_dir}[/cyan]")
        else:
            console.print(f":file_folder: Using existing FAISS index from [cyan]{faiss_index_file}[/cyan] and map from [cyan]{faiss_metadata_file}[/cyan]")
        progress.update(task_faiss, completed=1, description="FAISS index created.")

    console.print("\n:rocket: [bold green]Data pipeline ready![/bold green]")
    return faiss_index_file, faiss_metadata_file, processed_chunks_file

def perform_query(query: str):
    console.print(f"\n:search_left: Searching for: '{query}'")
    start_time = time.time()

    # Load necessary components
    console.print("Loading FAISS index and metadata map...")
    faiss_idx, chunk_meta_list = load_faiss_index()
    if not faiss_idx or not chunk_meta_list:
        console.print("[bold red]Could not load FAISS index or metadata. Build the index first.[/bold red]")
        return

    console.print("Loading embedding model for query...")
    sbert_model = get_sbert_model()
    console.print("Loading reranker model...")
    reranker = get_reranker_model()

    # Retrieve and rerank
    results = retrieve_and_rerank(
        query_text=query,
        faiss_index=faiss_idx,
        embedding_model=sbert_model,
        all_chunk_metadata=chunk_meta_list,
        reranker_model=reranker,
        top_k=config.faiss.top_k_retrieval,
        top_n=config.faiss.top_n_rerank
    )
    end_time = time.time()

    if results:
        console.print(f"\n:sparkles: Found {len(results)} relevant chunks in {end_time - start_time:.2f} seconds:")
        table = Table(title=f"Top {len(results)} Results for '{query[:50]}...'")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Chunk ID", style="cyan", no_wrap=True)
        table.add_column("Speaker", style="yellow")
        table.add_column("Text Snippet", style="green")
        table.add_column("Rerank Score", style="magenta", justify="right")

        for i, res in enumerate(results):
            snippet = Text(res.get('text', ''))
            snippet.truncate(150, overflow="ellipsis")
            table.add_row(
                str(i + 1),
                res.get('chunk_id', 'N/A'),
                res.get('speaker', 'N/A'),
                snippet,
                f"{res.get('rerank_score', 0.0):.4f}"
            )
        console.print(table)
    else:
        console.print(Panel.fit(f"[yellow]No results found for '{query}'.[/yellow]", 
                                title="Search Result", border_style="dim"))

def main():
    parser = argparse.ArgumentParser(description="ParlaMint RAG CLI - Build index and query.")
    parser.add_argument(
        "action", 
        choices=["build", "query"], 
        help="'build' to process data and create index, 'query' to ask questions."
    )
    parser.add_argument(
        "-q", "--query", 
        type=str, 
        help="The query string, required if action is 'query'."
    )
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=0, 
        help="Maximum number of raw TEI files to process during 'build' (0 for all). Useful for testing."
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force reprocessing of all data stages (chunking, embedding, indexing) even if files exist."
    )

    args = parser.parse_args()

    console.rule("[bold blue]ParlaMint RAG System[/bold blue]")

    if args.action == "build":
        console.print(f"Starting build process... Max raw files: {'all' if args.max_files == 0 else args.max_files}, Force reprocess: {args.force_reprocess}")
        ensure_data_pipeline(max_files_raw_data=args.max_files, force_reprocess=args.force_reprocess)
    
    elif args.action == "query":
        if not args.query:
            console.print("[bold red]Error: --query TEXT is required for the 'query' action.[/bold red]")
            parser.print_help()
            return
        perform_query(args.query)
    
    console.rule("[bold blue]Done.[/bold blue]")

if __name__ == "__main__":
    # Before running, ensure dummy files can be created by dependent scripts if they haven't run.
    # Example: python -m parlamint_rag.src.data_loader (to create dummy TEI)
    #          python -m parlamint_rag.src.text_processor (to create dummy chunks.jsonl)
    #          python -m parlamint_rag.src.embedding_generator (to create dummy embeddings)
    #          python -m parlamint_rag.src.faiss_indexer (to create dummy faiss index)
    # Then you can run:
    # python -m parlamint_rag.src.main build --max_files 2 --force_reprocess
    # python -m parlamint_rag.src.main query -q "example query"
    main() 