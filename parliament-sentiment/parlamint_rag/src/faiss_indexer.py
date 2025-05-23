#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAISS Indexing Module for ParlaMint RAG.

This module handles the creation, saving, and loading of FAISS (Facebook AI Similarity Search)
indexes. FAISS is used for efficient similarity searching of high-dimensional vectors (embeddings).

Main Components:
- `create_faiss_index()`: Builds a FAISS index from a NumPy array of embeddings, supporting
  various index types (e.g., IndexFlatL2, IndexFlatIP, HNSWFlat).
- `save_faiss_index()`: Saves the constructed FAISS index to disk, along with a list of
  metadata corresponding to each vector in the index.
- `load_faiss_index()`: Loads a previously saved FAISS index and its associated metadata map.

The module uses configurations from `config.py` for FAISS index types, filenames, and paths.
It typically consumes embeddings generated by `embedding_generator.py` and can be run
independently for testing its indexing and loading capabilities.
"""

import faiss
import numpy as np
from pathlib import Path
import pickle # For saving/loading the mapping from FAISS index to chunk_id
from typing import List, Dict, Any, Tuple

from .config import config
from .embedding_generator import load_embeddings_and_metadata # To get embeddings and metadata

def create_faiss_index(embeddings: np.ndarray, index_type: str = config.faiss.index_type) -> faiss.Index:
    """Creates a FAISS index of the specified type."""
    if embeddings.size == 0:
        raise ValueError("Cannot create FAISS index from empty embeddings.")
    if embeddings.dtype != np.float32:
        print(f"Converting embeddings from {embeddings.dtype} to float32 for FAISS.")
        embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    index = None

    print(f"Creating FAISS index of type '{index_type}' with dimension {dimension}...")

    if index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "IndexFlatIP": # Inner Product, useful for cosine similarity if vectors are normalized
        index = faiss.IndexFlatIP(dimension)
    elif index_type.startswith("IndexIVF"): # e.g., IndexIVFFlat, IndexIVFPQ
        # These require a training step on a subset of the data
        # For simplicity in this initial version, we'll require it to be pre-trained or use Flat
        # A more complete implementation would handle training here.
        # Example: nlist = 100 # Number of voronoi cells
        # quantizer = faiss.IndexFlatL2(dimension)
        # index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        # print("Training IVF index...")
        # index.train(embeddings) # Train on all or a subset of embeddings
        raise NotImplementedError(f"Index type '{index_type}' requires a training step. Use IndexFlatL2 or implement training.")
    elif index_type == "HNSWFlat":
        # HNSW (Hierarchical Navigable Small World) is good for speed and accuracy
        # M is the number of neighbors, efConstruction is for build-time quality/speed trade-off
        M = 32 # Typical value
        index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)
        index.hnsw.efConstruction = 40 # Higher is more accurate but slower to build
    else:
        raise ValueError(f"Unsupported FAISS index type: {index_type}")

    print(f"Adding {embeddings.shape[0]} vectors to the FAISS index...")
    index.add(embeddings)
    print(f"FAISS index created successfully. Total vectors in index: {index.ntotal}")
    return index

def save_faiss_index(
    index: faiss.Index, 
    chunk_metadata: List[Dict[str, Any]],
    output_dir: Path = Path(config.paths.index_dir),
    index_filename: str = config.faiss.index_filename,
    metadata_map_filename: str = config.faiss.metadata_filename # Was embedding_metadata.pkl, now faiss_metadata.pkl
) -> None:
    """Saves the FAISS index and a mapping from index ID to chunk metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / index_filename
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to {index_path}")

    # Save the metadata list directly. The index of a metadata item corresponds to its FAISS ID.
    metadata_map_path = output_dir / metadata_map_filename 
    with open(metadata_map_path, 'wb') as f_map:
        pickle.dump(chunk_metadata, f_map)
    print(f"FAISS metadata map (list of chunk_metadata) saved to {metadata_map_path}")

def load_faiss_index(
    index_dir: Path = Path(config.paths.index_dir),
    index_filename: str = config.faiss.index_filename,
    metadata_map_filename: str = config.faiss.metadata_filename
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Loads the FAISS index and the metadata map."""
    index_path = index_dir / index_filename
    metadata_map_path = index_dir / metadata_map_filename

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not metadata_map_path.exists():
        raise FileNotFoundError(f"FAISS metadata map file not found: {metadata_map_path}")

    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    print(f"FAISS index loaded. Total vectors: {index.ntotal}, Dimensions: {index.d}")

    print(f"Loading FAISS metadata map from {metadata_map_path}...")
    with open(metadata_map_path, 'rb') as f_map:
        chunk_metadata_list = pickle.load(f_map)
    print(f"FAISS metadata map loaded. Number of entries: {len(chunk_metadata_list)}")
    
    return index, chunk_metadata_list

if __name__ == '__main__':
    print("--- Testing faiss_indexer --- ")
    # This assumes 'test_embeddings.npy' and 'test_embedding_metadata.pkl' 
    # were created by embedding_generator.py in the configured index_dir.

    index_input_dir = Path(config.paths.index_dir)
    test_embeddings_file = "test_embeddings.npy"       # From embedding_generator test
    test_metadata_file = "test_embedding_metadata.pkl" # From embedding_generator test

    # 1. Load embeddings and their metadata
    print(f"Loading test embeddings and metadata from {index_input_dir}...")
    try:
        # We use the loader from embedding_generator as it loads the direct output of that script
        embeddings_np, chunk_meta = load_embeddings_and_metadata(
            input_dir=index_input_dir,
            embeddings_filename=test_embeddings_file,
            metadata_filename=test_metadata_file
        )
    except FileNotFoundError:
        print(f"Test embedding files ({test_embeddings_file}, {test_metadata_file}) not found in {index_input_dir}.")
        print("Please run embedding_generator.py's main test block first.")
        embeddings_np, chunk_meta = np.array([]), []

    if embeddings_np.size > 0 and chunk_meta:
        print(f"Loaded {embeddings_np.shape[0]} embeddings for index creation.")
        
        # 2. Create FAISS index
        # Using default IndexFlatL2 from config for testing
        try:
            faiss_index = create_faiss_index(embeddings_np) # Uses config.faiss.index_type

            # 3. Save FAISS index and the metadata list
            test_faiss_index_name = "test_parlamint.index"
            test_faiss_meta_name = "test_parlamint_metadata.pkl"
            save_faiss_index(faiss_index, 
                             chunk_meta, # Pass the original chunk_meta list
                             index_filename=test_faiss_index_name,
                             metadata_map_filename=test_faiss_meta_name)

            # 4. Load FAISS index and metadata map (optional test)
            print("\n--- Testing loading saved FAISS index and metadata map ---")
            loaded_index, loaded_meta_map = load_faiss_index(
                index_filename=test_faiss_index_name,
                metadata_map_filename=test_faiss_meta_name
            )

            if loaded_index and loaded_meta_map:
                print(f"Successfully loaded FAISS index with {loaded_index.ntotal} vectors.")
                print(f"Successfully loaded metadata map with {len(loaded_meta_map)} entries.")
                # print("First entry in loaded metadata map:", loaded_meta_map[0] if loaded_meta_map else "Empty")
                
                # Basic search test (optional)
                if loaded_index.ntotal > 0:
                    print("\n--- Performing a basic search test ---")
                    query_vector = embeddings_np[0:1].astype(np.float32) # Take the first embedding as a query
                    k = min(3, loaded_index.ntotal) # Search for top k results
                    distances, indices = loaded_index.search(query_vector, k)
                    print(f"Search for vector 0 (self-search), top {k} results:")
                    print(f"  Indices: {indices}")
                    print(f"  Distances: {distances}")
                    if indices.size > 0 and loaded_meta_map:
                        print(f"  Metadata for first result (index {indices[0][0]}): {loaded_meta_map[indices[0][0]]['chunk_id']}")
            else:
                print("Failed to load FAISS index or metadata map for test.")

        except Exception as e:
            print(f"An error occurred during FAISS indexing test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping FAISS indexer tests as no embeddings were loaded.") 