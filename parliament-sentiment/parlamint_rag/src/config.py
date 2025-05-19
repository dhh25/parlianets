#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Hub for the ParlaMint RAG Project.

This module defines and centralizes all configuration parameters for the RAG system,
including paths, model names, processing settings, and FAISS indexing parameters.
It uses Pydantic for data validation and to provide a clear structure for settings.

Main Components:
- PathsConfig: Defines all relevant directory and file paths.
- ModelConfig: Specifies the names/identifiers for embedding and cross-encoder models.
- ProcessingConfig: Contains parameters for text chunking (e.g., size, overlap).
- FaissConfig: Holds settings for FAISS index creation and retrieval.
- Config: A main Pydantic model aggregating all other configuration models.

A global `config` instance is exported for easy access throughout the project.
"""

# Configuration for the ParlaMint RAG project

from pydantic import BaseModel, DirectoryPath, FilePath
from typing import Optional

class PathsConfig(BaseModel):
    data_dir: DirectoryPath = "data"
    raw_data_dir: DirectoryPath = "data/raw"
    processed_data_dir: DirectoryPath = "data/processed"
    index_dir: DirectoryPath = "index"
    # log_dir: DirectoryPath = "logs"

class ModelConfig(BaseModel):
    embedding_model: str = "gemma3:4b"
    cross_encoder_model: str = "gemma3:4b"
    # llm_model: Optional[str] = None # For later integration

class ProcessingConfig(BaseModel):
    chunk_size: int = 300 # words
    chunk_overlap: int = 50 # words
    # Alternative: use token count with a tokenizer

class FaissConfig(BaseModel):
    index_type: str = "IndexFlatL2" # e.g., IndexFlatL2, IndexIVFFlat, HNSW
    index_filename: str = "parlamint_faiss.index"
    metadata_filename: str = "parlamint_metadata.pkl"
    top_k_retrieval: int = 50
    top_n_rerank: int = 5

class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    models: ModelConfig = ModelConfig()
    processing: ProcessingConfig = ProcessingConfig()
    faiss: FaissConfig = FaissConfig()

# Load and export a default config instance
config = Config()

if __name__ == "__main__":
    # Example of how to use the config
    print(f"Data directory: {config.paths.data_dir}")
    print(f"Embedding model: {config.models.embedding_model}")
    print(f"Chunk size: {config.processing.chunk_size}")
    # You can also create a config from a dictionary or JSON file if needed
    # external_config = {
    # "paths": {"data_dir": "/new/path/to/data"},
    # "models": {"embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"}
    # }
    # new_config = Config(**external_config)
    # print(f"New data_dir: {new_config.paths.data_dir}") 