#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding Generation Module for ParlaMint RAG.

This module is responsible for converting text chunks into numerical vector
representations (embeddings) using sentence-transformer models. These embeddings
are crucial for semantic search and similarity comparisons in the RAG pipeline.

Main Components:
- `get_embedding_model()`: Loads and caches a specified sentence-transformer model.
- `generate_embeddings()`: Takes a list of text chunks and produces a NumPy array of embeddings.
- `save_embeddings_and_metadata()`: Saves the generated embeddings and their corresponding
  metadata (e.g., chunk IDs, original source) to disk.
- `load_embeddings_and_metadata()`: Loads previously saved embeddings and metadata.

The module uses configurations from `config.py` for the embedding model name and paths.
It can be run independently for testing, typically using chunk data produced by `text_processor.py`.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path
import pickle # For saving metadata alongside embeddings if needed

from .config import config
from .text_processor import load_chunks_from_jsonl # To get chunks for embedding

# Global model cache to avoid reloading on multiple calls (if script is run multiple times)
# In a real application, you might manage models more sophisticatedly.
_model: SentenceTransformer | None = None # Type hint for clarity
_loaded_model_name: str | None = None # Stores the name of the currently loaded model

def get_embedding_model(model_name: str = config.models.embedding_model) -> SentenceTransformer:
    """Loads and returns the sentence transformer model."""
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        print(f"Loading embedding model: {model_name}")
        try:
            _model = SentenceTransformer(model_name)
            _loaded_model_name = model_name # NEW: Store the name of the loaded model
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
    return _model

def generate_embeddings(chunks: List[Dict[str, Any]], 
                        model_name: str = config.models.embedding_model,
                        batch_size: int = 32) -> np.ndarray:
    """Generates embeddings for a list of text chunks."""
    if not chunks:
        print("No chunks provided for embedding generation.")
        return np.array([])

    model = get_embedding_model(model_name)
    
    texts_to_embed = [chunk['text'] for chunk in chunks]
    print(f"Generating embeddings for {len(texts_to_embed)} chunks using model {model_name}...")
    
    # .encode can take a list of sentences and a batch_size argument
    embeddings = model.encode(texts_to_embed, 
                              batch_size=batch_size, 
                              show_progress_bar=True)
    
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}.")
    return embeddings

def save_embeddings_and_metadata(embeddings: np.ndarray, 
                                 chunk_metadata: List[Dict[str, Any]], 
                                 output_dir: Path = Path(config.paths.index_dir),
                                 embeddings_filename: str = "embeddings.npy",
                                 metadata_filename: str = "embedding_metadata.pkl") -> None:
    """Saves embeddings (NumPy array) and corresponding chunk metadata (list of dicts)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = output_dir / embeddings_filename
    np.save(embeddings_path, embeddings)
    print(f"Embeddings saved to {embeddings_path}")

    metadata_path = output_dir / metadata_filename
    with open(metadata_path, 'wb') as f_meta:
        pickle.dump(chunk_metadata, f_meta)
    print(f"Chunk metadata for embeddings saved to {metadata_path}")

def load_embeddings_and_metadata(input_dir: Path = Path(config.paths.index_dir),
                                 embeddings_filename: str = "embeddings.npy",
                                 metadata_filename: str = "embedding_metadata.pkl") -> (np.ndarray, List[Dict[str, Any]]):
    """Loads embeddings and corresponding chunk metadata."""
    embeddings_path = input_dir / embeddings_filename
    metadata_path = input_dir / metadata_filename

    if not embeddings_path.exists() or not metadata_path.exists():
        print(f"Error: Embeddings or metadata file not found in {input_dir}.")
        print(f"Checked for: {embeddings_path}, {metadata_path}")
        return np.array([]), []

    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings from {embeddings_path}")

    with open(metadata_path, 'rb') as f_meta:
        chunk_metadata = pickle.load(f_meta)
    print(f"Loaded chunk metadata from {metadata_path}")
    
    return embeddings, chunk_metadata


if __name__ == '__main__':
    print("--- Testing embedding_generator --- ")
    # This assumes 'test_chunks.jsonl' was created by text_processor.py in data/processed/
    # If not, run text_processor.py first.
    
    processed_dir = Path(config.paths.processed_data_dir)
    input_jsonl_file = processed_dir / "test_chunks.jsonl"

    if not input_jsonl_file.exists():
        print(f"Chunk file {input_jsonl_file} not found. Please run text_processor.py first.")
        print("Skipping embedding generation tests.")
    else:
        # 1. Load chunks
        print(f"Loading chunks from {input_jsonl_file}...")
        # Convert generator to list for this test script
        # For large datasets, process in batches/stream if memory is a concern
        chunks_to_embed = list(load_chunks_from_jsonl(input_jsonl_file)) 
        
        if not chunks_to_embed:
            print("No chunks loaded. Skipping embedding generation.")
        else:
            print(f"Loaded {len(chunks_to_embed)} chunks for embedding.")
            # For testing, let's only embed a small number of chunks to be quick
            # and to avoid downloading a large model immediately if not present.
            # We use a known small and fast model for this test.
            test_model_name = "sentence-transformers/all-MiniLM-L6-v2" # Smaller, faster model for testing
            print(f"Using test model: {test_model_name}")
            
            # Take a small sample for testing
            sample_chunks = chunks_to_embed[:5] 
            print(f"Processing a sample of {len(sample_chunks)} chunks for the test.")

            # 2. Generate embeddings
            # Use a small batch size for the sample data
            generated_embeddings = generate_embeddings(sample_chunks, model_name=test_model_name, batch_size=2)

            if generated_embeddings.size > 0:
                print(f"Shape of generated embeddings: {generated_embeddings.shape}")
                
                # Extract corresponding metadata for the sample
                sample_metadata = [chunk for chunk in sample_chunks] 

                # 3. Save embeddings and metadata
                index_output_dir = Path(config.paths.index_dir)
                test_embeddings_file = "test_embeddings.npy"
                test_metadata_file = "test_embedding_metadata.pkl"
                save_embeddings_and_metadata(generated_embeddings, 
                                             sample_metadata, 
                                             output_dir=index_output_dir, 
                                             embeddings_filename=test_embeddings_file,
                                             metadata_filename=test_metadata_file)

                # 4. Load them back (optional test)
                print("\n--- Testing loading saved embeddings and metadata ---")
                loaded_embeddings, loaded_metadata = load_embeddings_and_metadata(
                    input_dir=index_output_dir,
                    embeddings_filename=test_embeddings_file,
                    metadata_filename=test_metadata_file
                )
                if loaded_embeddings.size > 0 and loaded_metadata:
                    print(f"Successfully loaded embeddings of shape: {loaded_embeddings.shape}")
                    print(f"Successfully loaded {len(loaded_metadata)} metadata entries.")
                    # print("First loaded metadata entry:", loaded_metadata[0])
                else:
                    print("Failed to load embeddings or metadata for test.")
            else:
                print("No embeddings were generated for the sample.") 