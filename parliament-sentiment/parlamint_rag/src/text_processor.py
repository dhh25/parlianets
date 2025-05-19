#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text Processing Module for ParlaMint RAG.

This module handles the processing of text data, primarily focusing on dividing
raw text (utterances from ParlaMint TEI files) into smaller, manageable chunks.
These chunks are then used for generating embeddings.

Main Components:
- `split_text_into_words()`: A simple utility to split text by whitespace (can be enhanced).
- `create_chunks()`: Core function to segment documents into chunks based on word count
  and overlap, preserving relevant metadata.
- `save_chunks_to_jsonl()`: Saves the processed chunks to a JSONL file for persistence.
- `load_chunks_from_jsonl()`: Loads chunks iteratively from a JSONL file.

The module uses configurations from `config.py` for chunk size and overlap and can be
run independently for testing its functionality, using dummy data created by `data_loader.py`.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Generator
from tqdm import tqdm

from .config import config
from .data_loader import load_parlamint_data # To get data for chunking

# A simple word-based splitter, can be replaced with token-based later
def split_text_into_words(text: str) -> List[str]:
    """Splits text into words based on whitespace."""
    return text.split()

def create_chunks(documents: List[Dict[str, Any]], 
                  chunk_size: int = config.processing.chunk_size, 
                  chunk_overlap: int = config.processing.chunk_overlap,
                  id_prefix: str = "chunk") -> List[Dict[str, Any]]:
    """Chunks documents into smaller pieces with overlap."""
    all_chunks = []
    chunk_id_counter = 0

    for doc in tqdm(documents, desc="Chunking documents"):
        text = doc.get('text', '')
        if not text:
            continue

        words = split_text_into_words(text)
        if not words:
            continue

        start_index = 0
        while start_index < len(words):
            end_index = min(start_index + chunk_size, len(words))
            chunk_text = " ".join(words[start_index:end_index])
            chunk_id_counter += 1
            
            current_chunk_id = f"{doc.get('utterance_id', doc.get('doc_id', 'unknown'))}_{id_prefix}{chunk_id_counter}"

            chunk_data = {
                "chunk_id": current_chunk_id,
                "text": chunk_text,
                "original_doc_id": doc.get('doc_id'),
                "original_utterance_id": doc.get('utterance_id'),
                "speaker": doc.get('speaker'),
                "source_file": doc.get('source_file'),
                "word_count": len(words[start_index:end_index]),
                "char_count": len(chunk_text)
            }
            all_chunks.append(chunk_data)

            if end_index == len(words): # Reached the end of the document
                break
            
            # Move start_index for the next chunk, considering overlap
            start_index += (chunk_size - chunk_overlap)
            if start_index >= len(words): # Ensure we don't create empty or too small trailing chunks due to overlap logic
                break
            # Prevent re-processing the same small end piece if overlap is large
            if start_index > end_index - chunk_overlap and end_index < len(words):
                 start_index = end_index - chunk_overlap if end_index - chunk_overlap > 0 else 0

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents.")
    return all_chunks

def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_file: Path) -> None:
    """Saves a list of chunks to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in tqdm(chunks, desc=f"Saving chunks to {output_file.name}"):
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Successfully saved {len(chunks)} chunks to {output_file}")

def load_chunks_from_jsonl(input_file: Path) -> Generator[Dict[str, Any], None, None]:
    """Loads chunks from a JSONL file iteratively."""
    if not input_file.exists():
        print(f"Error: Chunk file {input_file} not found.")
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading chunks from {input_file.name}"):
            yield json.loads(line)

if __name__ == '__main__':
    # This example assumes dummy TEI files were created by data_loader.py when it was run.
    # If not, run data_loader.py first or ensure dummy files exist in config.paths.raw_data_dir
    
    print("--- Testing text_processor --- ")
    # 1. Load data using data_loader
    # Using a small number of files for testing
    raw_documents = load_parlamint_data(max_files=2) 

    if not raw_documents:
        print("No documents loaded. Ensure dummy TEI files exist in data/raw or run data_loader.py.")
        print("Skipping chunking tests.")
    else:
        # 2. Create chunks
        # Using smaller chunk size for testing visibility
        test_chunk_size = 15
        test_chunk_overlap = 5
        print(f"Using test chunk_size={test_chunk_size}, overlap={test_chunk_overlap}")
        chunks = create_chunks(raw_documents, 
                               chunk_size=test_chunk_size, 
                               chunk_overlap=test_chunk_overlap)

        if chunks:
            print(f"\nFirst 3 created chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"Chunk {i+1}:")
                print(json.dumps(chunk, indent=2, ensure_ascii=False))
                print("---")

            # 3. Save chunks to JSONL
            processed_dir = Path(config.paths.processed_data_dir)
            output_jsonl_file = processed_dir / "test_chunks.jsonl"
            save_chunks_to_jsonl(chunks, output_jsonl_file)

            # 4. Load chunks from JSONL (as a generator)
            print(f"\nLoading chunks from {output_jsonl_file} (first 3):")
            loaded_chunks_gen = load_chunks_from_jsonl(output_jsonl_file)
            for i, chunk in enumerate(loaded_chunks_gen):
                if i < 3:
                    print(f"Loaded chunk {i+1}:")
                    print(json.dumps(chunk, indent=2, ensure_ascii=False))
                    print("---")
                else:
                    break # Only print a few for brevity
            # Count total loaded chunks
            count = 0
            for _ in load_chunks_from_jsonl(output_jsonl_file): count+=1 # consume generator
            print(f"Total chunks loaded from file: {count}")
        else:
            print("No chunks were created.") 