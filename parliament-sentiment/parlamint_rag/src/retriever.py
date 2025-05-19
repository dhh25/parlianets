#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document Retrieval Module for ParlaMint RAG.

This module implements the core retrieval logic, which involves a multi-stage process
to find the most relevant text chunks for a given user query.

Main Components:
- `get_reranker_model()`: Loads and caches a CrossEncoder model used for re-ranking initial search results.
- `retrieve_and_rerank()`: Orchestrates the retrieval process:
    1. Embeds the user query using a sentence-transformer model.
    2. Performs an initial, fast search against a FAISS index to get a set of candidate chunks (top-k).
    3. Re-ranks these candidates using a more powerful (but slower) CrossEncoder model to refine
       the relevance scores.
    4. Returns the final top-n most relevant chunks with their metadata.

Relies on `embedding_generator.py` for query embedding and `faiss_indexer.py` for loading
the FAISS index and metadata. Configurations for models and retrieval parameters are sourced
from `config.py`.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any, Tuple

from .config import config
from .embedding_generator import get_embedding_model # For query embedding
from .faiss_indexer import load_faiss_index # For initial retrieval

# Global model caches
_reranker_model: CrossEncoder | None = None # Type hint for clarity
_loaded_reranker_model_name: str | None = None # Stores the name of the currently loaded model

def get_reranker_model(model_name: str = config.models.cross_encoder_model) -> CrossEncoder:
    """Loads and returns the CrossEncoder model for re-ranking."""
    global _reranker_model, _loaded_reranker_model_name
    if _reranker_model is None or _loaded_reranker_model_name != model_name:
        print(f"Loading reranker model: {model_name}")
        try:
            _reranker_model = CrossEncoder(model_name)
            _loaded_reranker_model_name = model_name
            print(f"Reranker model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading reranker model {model_name}: {e}")
            raise
    return _reranker_model

def retrieve_and_rerank(
    query_text: str,
    faiss_index: faiss.Index,
    embedding_model: SentenceTransformer,
    all_chunk_metadata: List[Dict[str, Any]], # This is the list of metadata loaded with FAISS index
    reranker_model: CrossEncoder,
    top_k: int = config.faiss.top_k_retrieval, # Initial retrieval count
    top_n: int = config.faiss.top_n_rerank    # Final count after reranking
) -> List[Dict[str, Any]]:
    """Performs multi-stage retrieval: FAISS search then CrossEncoder re-ranking."""
    if not query_text:
        print("Query text is empty. Cannot retrieve.")
        return []

    # 1. Embed the query
    print(f"Embedding query: '{query_text[:100]}...'")
    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    
    # 2. Initial retrieval from FAISS
    print(f"Performing FAISS search for top {top_k} candidates...")
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    retrieved_docs_initial = []
    if indices.size == 0 or indices[0][0] == -1: # -1 indicates no result or error
        print("No documents found in initial FAISS search.")
        return []

    for i in range(indices.shape[1]):
        idx = indices[0][i]
        if idx < 0 or idx >= len(all_chunk_metadata): # Check bounds
            # print(f"Warning: FAISS index {idx} out of bounds for metadata list (size {len(all_chunk_metadata)}). Skipping.")
            continue
        doc_metadata = all_chunk_metadata[idx]
        doc_metadata['faiss_distance'] = float(distances[0][i])
        doc_metadata['faiss_retrieval_rank'] = i + 1
        retrieved_docs_initial.append(doc_metadata)
    
    if not retrieved_docs_initial:
        print("No valid documents after filtering FAISS results.")
        return []
    print(f"Retrieved {len(retrieved_docs_initial)} candidates from FAISS.")

    # 3. Re-ranking with CrossEncoder
    print(f"Re-ranking top {len(retrieved_docs_initial)} candidates with {config.models.cross_encoder_model}...")
    cross_inp = [[query_text, doc['text']] for doc in retrieved_docs_initial]
    
    try:
        cross_scores = reranker_model.predict(cross_inp, show_progress_bar=True)
    except Exception as e:
        print(f"Error during reranking prediction: {e}")
        # Fallback to FAISS results if reranker fails
        return sorted(retrieved_docs_initial, key=lambda x: x['faiss_distance'])[:top_n]

    for i, doc in enumerate(retrieved_docs_initial):
        doc['rerank_score'] = float(cross_scores[i])
    
    # Sort by rerank_score in descending order (higher is better for most CrossEncoders)
    reranked_docs = sorted(retrieved_docs_initial, key=lambda x: x['rerank_score'], reverse=True)
    
    final_results = reranked_docs[:top_n]
    print(f"Re-ranking complete. Returning top {len(final_results)} documents.")
    
    return final_results

if __name__ == '__main__':
    print("--- Testing retriever --- ")
    # This test assumes a FAISS index and metadata map have been created and saved by faiss_indexer.py
    # specifically using 'test_parlamint.index' and 'test_parlamint_metadata.pkl'

    try:
        print("Loading FAISS index and metadata for retriever test...")
        # Use the test index names specified in faiss_indexer's __main__ block
        faiss_index_instance, all_chunk_metadata_list = load_faiss_index(
            index_filename="test_parlamint.index",
            metadata_map_filename="test_parlamint_metadata.pkl"
        )
        
        if not faiss_index_instance or not all_chunk_metadata_list:
            raise FileNotFoundError("Test FAISS index or metadata not loaded.")

        print("Loading embedding model for query encoding...")
        # Use the same (or compatible) embedding model used for creating the index
        # For testing, using the small model from embedding_generator test.
        # The actual config.models.embedding_model might be different.
        query_embed_model = get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2") 

        print("Loading reranker model...")
        # Using a small reranker for testing, if available or a default one.
        # The actual config.models.cross_encoder_model might be different and larger.
        try:
            # Attempt to load a smaller cross-encoder for testing if not already specified as such in config
            # For a robust test, ensure a small cross-encoder is available or use the one from config
            reranker = get_reranker_model(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2") 
        except Exception:
            print(f"Small test reranker not found, trying config default: {config.models.cross_encoder_model}")
            reranker = get_reranker_model() # Uses config.models.cross_encoder_model

        test_query = "mitä mieltä ministeri oli asiasta"
        print(f"\nTest query: '{test_query}'")

        retrieved_results = retrieve_and_rerank(
            query_text=test_query,
            faiss_index=faiss_index_instance,
            embedding_model=query_embed_model,
            all_chunk_metadata=all_chunk_metadata_list,
            reranker_model=reranker,
            top_k=5, # Requesting fewer for test display
            top_n=2  # Requesting fewer for test display
        )

        if retrieved_results:
            print(f"\nTop {len(retrieved_results)} results for query '{test_query}':")
            for i, res in enumerate(retrieved_results):
                print(f"  Rank {i+1}:")
                print(f"    Chunk ID: {res.get('chunk_id')}")
                print(f"    Text: '{res.get('text', '')[:150]}...'")
                print(f"    Speaker: {res.get('speaker')}")
                print(f"    Source File: {res.get('source_file')}")
                print(f"    FAISS Distance: {res.get('faiss_distance'):.4f}")
                print(f"    Rerank Score: {res.get('rerank_score'):.4f}")
        else:
            print(f"No results retrieved for the query '{test_query}'.")

    except FileNotFoundError as e:
        print(f"File not found during retriever test setup: {e}")
        print("Please ensure that embedding_generator.py and faiss_indexer.py (main test blocks) have been run successfully first.")
    except Exception as e:
        print(f"An error occurred during retriever test: {e}")
        import traceback
        traceback.print_exc() 