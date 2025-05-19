#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Comparison Sentiment Analysis Tool

This script analyzes sentiment in text using Ollama models and compares their performance.
It processes CSV data containing text snippets and runs sentiment analysis (positive/negative/neutral)
and optionally topic modeling. The tool is optimized to work on different hardware environments
(Apple Silicon, NVIDIA GPUs, or CPU-only) and outputs results in both CSV and Markdown formats.

Author: Jussi Wright
Email: jussi.wright@helsinki.fi
"""

import csv
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import concurrent.futures
import time
import os
import platform
import argparse
from datetime import datetime

# --- System Environment Detection ---
def detect_system_environment():
    """Detect system environment details for optimizations."""
    env_info = {
        "system": platform.system(),        # 'Darwin', 'Linux', 'Windows'
        "processor": platform.processor(),  # 'arm', 'x86_64', etc.
        "is_apple_silicon": False,
        "has_nvidia_gpu": False,
        "ollama_path": None,
        "recommended_params": {}
    }
    
    # Check for Apple Silicon (M-series)
    if env_info["system"] == 'Darwin' and env_info["processor"] == 'arm':
        env_info["is_apple_silicon"] = True
        env_info["recommended_params"] = {
            "num_ctx": 8192,
            "num_gpu": 99,  # Uses all GPU memory on Apple Silicon
            "num_thread": 8,
            "use_mmap": True,
            "f16_kv": True,
        }
        env_info["ollama_path"] = "/opt/homebrew/bin/ollama" if os.path.exists("/opt/homebrew/bin/ollama") else "/usr/local/bin/ollama"
    
    # Check for Linux with NVIDIA GPU
    elif env_info["system"] == 'Linux':
        try:
            # Try to import CUDA-related packages for detection
            import subprocess
            nvidia_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if nvidia_output.returncode == 0:
                env_info["has_nvidia_gpu"] = True
                env_info["recommended_params"] = {
                    "num_ctx": 8192,
                    "num_gpu": 1,  # Typically using 1 GPU
                    "num_thread": 8,
                    "use_mmap": True,
                    "f16_kv": True,
                }
        except (FileNotFoundError, ImportError, subprocess.SubprocessError):
            # No NVIDIA GPU or nvidia-smi not available
            pass
        
        # Check for Ollama on Linux
        if os.path.exists("/usr/local/bin/ollama"):
            env_info["ollama_path"] = "/usr/local/bin/ollama"
        elif os.path.exists("/usr/bin/ollama"):
            env_info["ollama_path"] = "/usr/bin/ollama"
    
    # Windows (limited support)
    elif env_info["system"] == 'Windows':
        env_info["recommended_params"] = {
            "num_ctx": 8192,
            "num_thread": 8,
            "use_mmap": True,
        }
    
    return env_info

# --- Configuration ---
# Detect system environment
SYS_ENV = detect_system_environment()
IS_APPLE_SILICON = SYS_ENV["is_apple_silicon"]
HAS_NVIDIA_GPU = SYS_ENV["has_nvidia_gpu"]

# Ollama parameters based on detected environment
OLLAMA_PARAMS = SYS_ENV["recommended_params"]

# List your available OLLAMA models here
AVAILABLE_OLLAMA_MODELS: List[str] = [
    "gemma3:4b-it-qat", 
    "deepseek-r1:32b-8k",
    # Add other models as needed, for example:
    # "llama3.1:8b",
    # "gemma2:9b-it",
    # "gemma2:27b-it"
]

# Get the model name for filenames (takes the first model in the list)
MODEL_NAME_FOR_FILES = AVAILABLE_OLLAMA_MODELS[0].replace(":", "_")

# Path to the input CSV file (relative to this script's location)
CSV_FILE_PATH: Path = Path(__file__).parent / "data" / "LLM_test_dataset.csv"

OUTPUT_BASE_DIR: Path = Path(__file__).parent / "output" 
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MD_FILE_PATH: Path = Path(__file__).parent.parent / f"sentiment_analysis_results_{MODEL_NAME_FOR_FILES}.md"
OUTPUT_CSV_FILE_PATH: Path = Path(__file__).parent / f"sentiment_analysis_output_{MODEL_NAME_FOR_FILES}.csv"

ROWS_TO_PROCESS: int = 0 
CLASSIFICATION_TEMPERATURE: float = 0.0
OLLAMA_API_URL: str = "http://localhost:11434/api/generate"
API_TIMEOUT: int = 600
MAX_WORKERS: int = 4

# Default configuration (can be overridden by command-line arguments)
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_SENTIMENT_MODEL = "gemma3:4b-it-qat" # User specified default, changed
DEFAULT_TOPIC_MODEL = "gemma3:4b-it-qat" # User specified default
DEFAULT_CSV_FILE_PATH = "Parlament/sentiment_analyzer/raw_data/LLM_test_dataset.csv"
DEFAULT_OUTPUT_DIR = "Parlament/sentiment_analyzer/model_comparison/results_model_comparisons"

# Column names in the input CSV
TEXT_COLUMN_SENTIMENT = "Sentence"
TEXT_COLUMN_TOPIC = "Sentence_longer"
GROUND_TRUTH_COLUMN = "Ground Truth"
DOC_ID_COLUMN = "Doc ID"
# Add other original columns that should be carried over if they exist
ORIGINAL_COLUMNS_TO_CARRY = [DOC_ID_COLUMN, "Token ID", "Speech ID", "Target_country", "Sentence", "Sentence_longer", "Ground Truth", "Notes"]

# --- Helper Functions ---

def construct_prompt(sentence: str, country: str, context: str) -> str:
    """Constructs the prompt for the OLLAMA model with few-shot examples for better accuracy."""
    return f"""###
Your task is to classify the mention of a country in a parliamentary transcript as positive, neutral or negative.
- A positive mention is one that puts the target country in a positive light
- A neutral mention refers to factual, descriptive, or incidental references to the country that do not convey any strong sentiment.
- A negative mention is one that criticizes the country, its government, policies, or actions, or portrays it in an unfavorable light.

# Examples (different from the data you will classify):
1. "Spain's commitment to renewable energy has grown remarkably in recent years." (POSITIVE - praises Spain's progress)
2. "Portugal was one of several nations attending the conference last week." (NEUTRAL - factual mention of Portugal without sentiment)
3. "The delegation from Norway arrived yesterday for the talks." (NEUTRAL - simple factual statement)
4. "Australia's mining regulations have been too lax, resulting in environmental damage." (NEGATIVE - criticizes Australia's regulations)
5. "Taiwan has shown impressive technological advancement in semiconductor manufacturing." (POSITIVE - praises Taiwan's tech advancements)
6. "The Czech Republic's position on this matter remains unclear." (NEUTRAL - factual statement without judgment)
7. "Singapore's housing policies have created an inclusive model worth studying." (POSITIVE - positive mention of Singapore's approach)
8. "Denmark's social welfare system ensures no citizen is left behind." (POSITIVE - praises Denmark's system)
9. "South Africa was mentioned in the report on page 32." (NEUTRAL - purely referential)
10. "Brazil's deforestation rates are alarming and require immediate action." (NEGATIVE - criticizes Brazil's environmental situation)

# Sentence to classify:
{sentence}

# Target country:
{country}

# Sentence Context:
{context}

Analyze the sentiment expressed about the target country and classify it as POSITIVE, NEUTRAL, or NEGATIVE.

Output a JSON response with only the following format:
```json
{{
  "sentiment": "VALUE" // Where VALUE is exactly one of: POSITIVE, NEUTRAL, NEGATIVE
}}
```"""

def call_ollama_api(prompt: str, model_name: str, api_url: str, temperature: float = 0.0) -> tuple[str | None, str | None]:
    """
    Calls the Ollama API with the given prompt and model.
    Args:
        prompt: The prompt to send to the model.
        model_name: The name of the Ollama model to use.
        api_url: The URL of the Ollama API.
        temperature: The temperature for generation.
    Returns:
        A tuple containing (response_text, error_message).
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, **OLLAMA_PARAMS} if IS_APPLE_SILICON else {"temperature": temperature}
    }
    try:
        response = requests.post(api_url, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        api_response_text = result.get("response", "").strip()
        return api_response_text, None
    except requests.exceptions.Timeout:
        return None, f"Timeout calling Ollama API for model {model_name} after {API_TIMEOUT}s."
    except requests.exceptions.RequestException as e:
        return None, f"Error calling Ollama API for model {model_name}: {e}"
    except json.JSONDecodeError:
        # It's good to see the problematic response text if JSON decoding fails
        response_text_for_error = "N/A"
        try:
            response_text_for_error = response.text
        except: #pylint: disable=bare-except
            pass
        return None, f"Error decoding JSON response from Ollama API for model {model_name}. Response: {response_text_for_error}"

def process_item(
    item_input: Dict[str, Any], # Renamed to avoid confusion with item being built
    sentiment_model_name: Optional[str] = None, 
    topic_model_name: Optional[str] = None,
    run_sentiment: bool = False,
    run_topic: bool = False,
    ollama_api_url_override: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single item (row) with the specified models and analyses."""
    row_idx = item_input["row_idx"] # This is the 1-based index from preprocessing
    
    # Initialize current_row_results with all original data from the input item
    current_row_results = {**item_input} 
    # Add/initialize new fields
    current_row_results.update({
        "Row_Internal_Index": row_idx, # Keep track of the internal processing index
        "Sentence_Snippet_Generated": " ".join(item_input.get(TEXT_COLUMN_SENTIMENT, "").split()[:10]) + ("..." if len(item_input.get(TEXT_COLUMN_SENTIMENT, "").split()) > 10 else ""),
        "Sentiment_Processing_Time_s": None,
        "Topic_Processing_Time_s": None,
        "Processing_Error_Details": None
    })

    actual_ollama_api_url = ollama_api_url_override or OLLAMA_API_URL
    sentence_for_sentiment = item_input.get(TEXT_COLUMN_SENTIMENT, "")
    context_for_topic = item_input.get(TEXT_COLUMN_TOPIC, "")
    ground_truth = item_input.get(GROUND_TRUTH_COLUMN, "")

    print(f"--- Processing Row {row_idx} (Doc ID: {item_input.get(DOC_ID_COLUMN, 'N/A')}) ---")

    if run_sentiment and sentiment_model_name:
        if not sentence_for_sentiment or sentence_for_sentiment == "N/A":
            print(f"  Skipping sentiment for row {row_idx} due to missing sentence.")
            current_row_results[f"Sentiment ({sentiment_model_name.replace(':', '_')})"] = "N/A_MISSING_INPUT"
        else:
            print(f"  Sentiment analysis for row {row_idx} with model: {sentiment_model_name}...")
            sentiment_prompt_text = construct_sentiment_prompt(sentence_for_sentiment)
            sentiment_start_time = time.perf_counter()
            api_response_text, error = call_ollama_api(sentiment_prompt_text, sentiment_model_name, actual_ollama_api_url)
            sentiment_end_time = time.perf_counter()
            current_row_results["Sentiment_Processing_Time_s"] = sentiment_end_time - sentiment_start_time
            
            sentiment_col_name = f"Sentiment ({sentiment_model_name.replace(':', '_')})"
            sentiment_correct_col_name = f"{sentiment_col_name} Correct"

            if error:
                print(error)
                current_row_results[sentiment_col_name] = "Error_API"
                current_row_results["Processing_Error_Details"] = error
            elif api_response_text:
                sentiment = parse_sentiment_response(api_response_text, sentence_for_sentiment)
                current_row_results[sentiment_col_name] = sentiment
            else:
                current_row_results[sentiment_col_name] = "Error_NoResponse"

            # Calculate correctness if sentiment was processed
            if current_row_results[sentiment_col_name] not in ["Error_API", "Error_NoResponse", "N/A_MISSING_INPUT"]:
                is_correct = current_row_results[sentiment_col_name].upper() == ground_truth.upper() if ground_truth and ground_truth != "N/A" else "N/A"
                current_row_results[sentiment_correct_col_name] = "✓" if is_correct == True else ("✗" if is_correct == False else "N/A")
            else:
                 current_row_results[sentiment_correct_col_name] = "N/A_ERROR"

            print(f"    Row {row_idx}: Model '{sentiment_model_name}' sentiment: {current_row_results.get(sentiment_col_name)} (Correct: {current_row_results.get(sentiment_correct_col_name)}) Time: {current_row_results['Sentiment_Processing_Time_s']:.3f}s")

    if run_topic and topic_model_name:
        if not context_for_topic or context_for_topic == "N/A":
            print(f"  Skipping topic for row {row_idx} due to missing context (Sentence_longer).")
            current_row_results[f"Topic Keyword ({topic_model_name.replace(':', '_')})"] = "N/A_MISSING_INPUT"
        else:
            print(f"  Topic modeling for row {row_idx} with model: {topic_model_name}...")
            topic_prompt_text = construct_topic_prompt(context_for_topic)
            topic_start_time = time.perf_counter()
            api_response_text, error = call_ollama_api(topic_prompt_text, topic_model_name, actual_ollama_api_url)
            topic_end_time = time.perf_counter()
            current_row_results["Topic_Processing_Time_s"] = topic_end_time - topic_start_time

            topic_col_name = f"Topic Keyword ({topic_model_name.replace(':', '_')})"
            if error:
                print(error)
                current_row_results[topic_col_name] = "Error_API"
                if current_row_results["Processing_Error_Details"]: current_row_results["Processing_Error_Details"] += f"; {error}" 
                else: current_row_results["Processing_Error_Details"] = error
            elif api_response_text:
                topic_keyword = parse_topic_response(api_response_text, context_for_topic)
                current_row_results[topic_col_name] = topic_keyword
            else:
                current_row_results[topic_col_name] = "Error_NoResponse"
            print(f"    Row {row_idx}: Model '{topic_model_name}' topic: {current_row_results.get(topic_col_name)} Time: {current_row_results['Topic_Processing_Time_s']:.3f}s")

    print(f"--- Finished Row {row_idx} ---\\n")
    return current_row_results

def calculate_metrics_manually(results_list: List[Dict[str, Any]], model_column_name: str, ground_truth_column_key: str = "Ground Truth"):
    """Calculates accuracy and other metrics manually from a list of result dictionaries."""
    print(f"\n--- Metrics for Model: {model_column_name} (Manual Calculation) ---")
    
    references_list = [str(row.get(ground_truth_column_key, "")).upper() for row in results_list]
    predictions_list = [str(row.get(model_column_name, "")).upper() for row in results_list]

    valid_indices = [
        i for i, (ref, pred) in enumerate(zip(references_list, predictions_list)) 
        if ref not in ["", "N/A"] and pred not in ["", "N/A", "ERROR_API", "ERROR_NORESPONSE", "UNKNOWN", "N/A_MISSING_INPUT", "N/A_ERROR"]
    ]
    
    if not valid_indices:
        print("  No valid predictions found to calculate metrics.")
        return {"model": model_column_name, "accuracy": 0, "valid_predictions": 0, "correct_predictions": 0, "class_metrics": {}}
    
    references = [references_list[i] for i in valid_indices]
    predictions = [predictions_list[i] for i in valid_indices]
    
    valid_count = len(references)
    correct_predictions = sum(1 for r, p in zip(references, predictions) if r == p)
    accuracy = correct_predictions / valid_count if valid_count > 0 else 0
    
    classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    class_metrics = {}
    
    for cls in classes:
        tp = sum(1 for r, p in zip(references, predictions) if p == cls and r == cls)
        fp = sum(1 for r, p in zip(references, predictions) if p == cls and r != cls)
        fn = sum(1 for r, p in zip(references, predictions) if p != cls and r == cls)
        support = sum(1 for r in references if r == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    
    print(f"  Accuracy: {accuracy:.4f} ({correct_predictions} correct out of {valid_count} valid predictions)")
    for cls, metrics in class_metrics.items():
        print(f"  {cls}: Support={metrics['support']}, P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    return {
        "model": model_column_name,
        "accuracy": accuracy,
        "valid_predictions": valid_count,
        "correct_predictions": correct_predictions,
        "class_metrics": class_metrics
    }

def preprocess_data(csv_file_path_str: str, rows_to_process_limit: int, cli_args: argparse.Namespace):
    """Loads and preprocesses the dataset, returning it as a list of items to process."""
    csv_file = Path(csv_file_path_str)
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return []
        
    items_to_process = []
    with open(csv_file, 'r', encoding='utf-8') as f_csv:
        reader = csv.DictReader(f_csv)
        if not reader.fieldnames:
            print(f"Error: CSV file {csv_file} is empty or has no header.")
            return []

        # Check for essential columns based on what will be run
        required_cols_for_sentiment = {TEXT_COLUMN_SENTIMENT, GROUND_TRUTH_COLUMN, DOC_ID_COLUMN, "Target_country"} if cli_args.run_sentiment else set()
        required_cols_for_topic = {TEXT_COLUMN_TOPIC, DOC_ID_COLUMN, "Target_country"} if cli_args.run_topic else set()
        all_required_cols = list(required_cols_for_sentiment.union(required_cols_for_topic))
        
        missing_cols = [col for col in all_required_cols if col not in reader.fieldnames]
        if missing_cols:
            print(f"Error: CSV file {csv_file} must contain columns: {', '.join(missing_cols)} for the selected operations.")
            return []
            
        for i, row_dict in enumerate(reader):
            if rows_to_process_limit > 0 and i >= rows_to_process_limit:
                print(f"Reached --rows limit of {rows_to_process_limit}.")
                break
            
            item = {"row_idx": i + 1} # 1-based index for processing
            for col_name in reader.fieldnames: # Carry all original columns
                item[col_name] = row_dict.get(col_name, "").strip()
            
            # Basic validation: skip if critical input is missing for selected tasks
            target_country_val = item.get("Target_country", "N/A").strip()
            sentence_val = item.get(TEXT_COLUMN_SENTIMENT, "N/A").strip()
            context_val = item.get(TEXT_COLUMN_TOPIC, "N/A").strip()
            doc_id_val = item.get(DOC_ID_COLUMN, str(i + 1)).strip()

            can_process_sentiment = cli_args.run_sentiment and sentence_val != "N/A" and target_country_val != "N/A"
            can_process_topic = cli_args.run_topic and context_val != "N/A" and target_country_val != "N/A"

            if not (can_process_sentiment or can_process_topic) and (cli_args.run_sentiment or cli_args.run_topic) :
                print(f"  Skipping CSV row {i+1} (Doc ID: {doc_id_val}) due to missing essential data for selected analyses (Sentence: '{sentence_val}', Context: '{context_val}', TargetCountry: '{target_country_val}').")
                continue
            
            items_to_process.append(item)
            
    if not items_to_process:
        print(f"No processable rows found in {csv_file} after initial filtering.")
    return items_to_process

def optimize_ollama_for_environment():
    """Optimizes Ollama based on the detected system environment."""
    print(f"Optimizing for {platform.system()} environment...")
    
    # Apple Silicon specific optimizations
    if IS_APPLE_SILICON:
        try:
            import torch
            if torch.backends.mps.is_available():
                 os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                 print("  Metal Performance Shaders (MPS) support detected and enabled for PyTorch.")
            else: print("  MPS not available for PyTorch.")
        except ImportError: print("  PyTorch not installed, skipping MPS check.")
        except AttributeError: print("  Error checking MPS availability.")
    
    # NVIDIA GPU specific optimizations for Linux
    elif HAS_NVIDIA_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  Using CUDA device: {torch.cuda.current_device()}")
            else:
                print("  CUDA is not available despite NVIDIA GPU being detected.")
        except ImportError:
            print("  PyTorch not installed, skipping CUDA check.")
    
    # Check Ollama service for all platforms
    try:
        ollama_check_url = (os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/tags").replace("///", "//")
        requests.get(ollama_check_url, timeout=2)
        print(f"  Ollama service detected as running at {ollama_check_url}.")
    except requests.exceptions.ConnectionError:
        print(f"  Warning: Ollama service not detected at {ollama_check_url}. Please ensure Ollama is running.")
    except Exception as e:
        print(f"  Warning: Could not check Ollama service status: {e}")

    # Check for Ollama binary
    if SYS_ENV["ollama_path"]:
        print(f"  Found Ollama installation at: {SYS_ENV['ollama_path']}")
    else:
        print("  Warning: Could not detect Ollama installation path.")
    
    print(f"  Using Ollama parameters: {OLLAMA_PARAMS}")
    print("Optimization checks complete\n")

def construct_sentiment_prompt(text: str) -> str:
    """Constructs the prompt for sentiment analysis."""
    return f"""Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral.
Return only the sentiment label (Positive, Negative, or Neutral).

Text: "{text}"

Sentiment:"""

def construct_topic_prompt(text: str) -> str:
    """Constructs the prompt for topic modeling (single keyword)."""
    return f"""Analyze the following text and identify a single, most representative keyword or short keyphrase (2-3 words max) that best describes its main topic.
Return only the keyword/keyphrase.

Text: "{text}"

Keyword:"""

def parse_sentiment_response(api_response: str, text_snippet: str) -> str:
    """Parses the sentiment from the Ollama API response."""
    cleaned_response = api_response.replace("```", "").replace('"', '').replace("'", "").strip().capitalize()
    if cleaned_response in ["Positive", "Negative", "Neutral"]:
        return cleaned_response
    lower_response = api_response.lower()
    if "positive" in lower_response: return "Positive"
    if "negative" in lower_response: return "Negative"
    if "neutral" in lower_response: return "Neutral"
    print(f"Warning: Model returned an unparsable sentiment: '{api_response}' for text: '{text_snippet[:50]}...'")
    return "Unknown"

def parse_topic_response(api_response: str, text_snippet: str) -> str:
    """Parses the topic keyword from the Ollama API response."""
    cleaned_response = api_response.replace("```", "").strip()
    if not cleaned_response:
        print(f"Warning: Model returned an empty topic keyword for text: '{text_snippet[:50]}...'")
        return "Unknown"
    return cleaned_response

def main():
    parser = argparse.ArgumentParser(description="Run sentiment and/or topic analysis on a CSV file using Ollama.")
    parser.add_argument("--csv_file", type=str, default=str(DEFAULT_CSV_FILE_PATH), help=f"Path to the input CSV file. Default: {DEFAULT_CSV_FILE_PATH}")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help=f"Directory to save the output files. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--run_sentiment", action=argparse.BooleanOptionalAction, default=None, help="Run sentiment analysis. Defaults to True if no run flags are given.")
    parser.add_argument("--run_topic", action=argparse.BooleanOptionalAction, default=False, help="Run topic modeling.")
    parser.add_argument("--sentiment_model", type=str, default=DEFAULT_SENTIMENT_MODEL, help=f"Ollama model for sentiment analysis. Default: {DEFAULT_SENTIMENT_MODEL}")
    parser.add_argument("--topic_model", type=str, default=DEFAULT_TOPIC_MODEL, help=f"Ollama model for topic modeling. Default: {DEFAULT_TOPIC_MODEL}")
    parser.add_argument("--ollama_api_url", type=str, default=OLLAMA_API_URL, help=f"Ollama API URL. Default: {OLLAMA_API_URL}")
    parser.add_argument("--rows", type=int, default=ROWS_TO_PROCESS, help=f"Number of rows to process (0 for all). Default: {ROWS_TO_PROCESS}")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS, help=f"Max concurrent workers for API calls. Default: {MAX_WORKERS}")

    args = parser.parse_args()

    if args.run_sentiment is None and not args.run_topic:
        args.run_sentiment = True
    elif args.run_sentiment is None: 
        args.run_sentiment = False

    if not args.run_sentiment and not args.run_topic:
        print("Neither sentiment analysis nor topic modeling selected to run. Exiting.")
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    run_start_datetime = datetime.now()
    timestamp = run_start_datetime.strftime("%Y%m%d_%H%M%S")
    base_filename_parts = ["results"]
    if args.run_sentiment:
        base_filename_parts.append(f"sentiment_{args.sentiment_model.replace(':', '_')}")
    if args.run_topic:
        base_filename_parts.append(f"topic_{args.topic_model.replace(':', '_')}")
    
    base_filename = "_".join(base_filename_parts) + f"_{timestamp}"
    output_csv_path = output_path / f"{base_filename}.csv"
    output_md_path = output_path / f"{base_filename}.md"

    # --- Print run configuration --- 
    print(f"Starting analysis run at: {run_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input CSV: {args.csv_file}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Output MD: {output_md_path}")
    if args.run_sentiment:
        print(f"Sentiment Analysis: True, Model: {args.sentiment_model}")
    if args.run_topic:
        print(f"Topic Modeling: True, Model: {args.topic_model}")
    print(f"Ollama API URL: {args.ollama_api_url}")
    print(f"Max workers: {args.max_workers}")
    
    if IS_APPLE_SILICON or HAS_NVIDIA_GPU:
        optimize_ollama_for_environment()
    else:
        print(f"Running on {platform.system()} ({platform.processor()}) without specific hardware optimizations.")

    items_to_process = preprocess_data(args.csv_file, args.rows, args)
    if not items_to_process:
        print("No items to process from CSV after preprocessing. Exiting.")
        return
    
    total_rows_in_input_list = len(items_to_process)
    print(f"Prepared {total_rows_in_input_list} items for processing.")

    all_results_data: List[Dict[str, Any]] = []
    overall_start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_item, 
                item, 
                args.sentiment_model if args.run_sentiment else None, 
                args.topic_model if args.run_topic else None,
                args.run_sentiment,
                args.run_topic,
                args.ollama_api_url 
            ) for item in items_to_process
        ]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                all_results_data.append(result)
                print(f"Completed processing for item {i+1}/{total_rows_in_input_list} (Original row index: {result.get('row_idx')})")
            except Exception as exc:
                # This should ideally capture errors from process_item if they are not caught inside
                print(f'An item generated an exception during concurrent execution: {exc}')
                # Optionally, append a placeholder or error dict to all_results_data if needed
    
    overall_end_time = time.perf_counter()
    total_processing_duration = overall_end_time - overall_start_time
    print(f"Finished processing all {total_rows_in_input_list} items in {total_processing_duration:.2f} seconds.")

    all_results_data.sort(key=lambda x: x.get('row_idx', float('inf'))) 

    # --- Output to CSV ---
    if all_results_data:
        # Dynamically determine column order for CSV
        # Start with original columns, then add generated ones
        final_ordered_columns = []
        if items_to_process: # Ensure there was at least one item to get original fieldnames
            # Attempt to get fieldnames from the first processed item, as it contains original fields
            # Fallback to an empty list if all_results_data is empty (though guarded by 'if all_results_data')
            first_result_keys = list(all_results_data[0].keys()) if all_results_data else []
            
            # Add original CSV columns that were carried over first, in their original order if possible
            # The `item_input` in `process_item` has these keys. `**item_input` copies them.
            # We can use ORIGINAL_COLUMNS_TO_CARRY as a guide for preferred order.
            for col in ORIGINAL_COLUMNS_TO_CARRY:
                if col in first_result_keys: # Check if it was actually in the input and thus in results
                    final_ordered_columns.append(col)
            
            # Add other columns from the first result that were not in ORIGINAL_COLUMNS_TO_CARRY (e.g. new ones added by preprocess_data)
            for col in first_result_keys:
                if col not in final_ordered_columns and col not in [ # Avoid re-adding dynamic model/time cols here
                    f"Sentiment ({args.sentiment_model.replace(':', '_')})", 
                    f"Sentiment ({args.sentiment_model.replace(':', '_')}) Correct", 
                    f"Topic Keyword ({args.topic_model.replace(':', '_')})",
                    "Sentiment_Processing_Time_s", "Topic_Processing_Time_s", "Processing_Error_Details",
                    "Sentence_Snippet_Generated", "Row_Internal_Index" # These will be added explicitly
                ]:
                    final_ordered_columns.append(col)
        
        # Add specific generated columns in a sensible order
        final_ordered_columns.append("Row_Internal_Index") # The 1-based index used during processing
        final_ordered_columns.append("Sentence_Snippet_Generated")

        if args.run_sentiment and args.sentiment_model:
            sm_name_formatted = args.sentiment_model.replace(':', '_')
            final_ordered_columns.append(f"Sentiment ({sm_name_formatted})")
            final_ordered_columns.append(f"Sentiment ({sm_name_formatted}) Correct")
            final_ordered_columns.append("Sentiment_Processing_Time_s")
        
        if args.run_topic and args.topic_model:
            tm_name_formatted = args.topic_model.replace(':', '_')
            final_ordered_columns.append(f"Topic Keyword ({tm_name_formatted})")
            final_ordered_columns.append("Topic_Processing_Time_s")
        
        final_ordered_columns.append("Processing_Error_Details")

        # Ensure all keys from results are included, even if missed by above logic (append at end)
        all_actual_keys = set()
        for res_dict in all_results_data:
            all_actual_keys.update(res_dict.keys())
        
        for key in sorted(list(all_actual_keys)):
            if key not in final_ordered_columns:
                final_ordered_columns.append(key)
        
        # Make sure no duplicates (though set logic should prevent it, defensive)
        final_ordered_columns = list(dict.fromkeys(final_ordered_columns))
        
        try:
            df_results = pd.DataFrame(all_results_data)
            # Filter to only include columns that actually exist in the DataFrame
            # This handles cases where a column might be defined in final_ordered_columns
            # but doesn't exist in any result dict (e.g., topic columns if topic was not run)
            existing_final_cols = [col for col in final_ordered_columns if col in df_results.columns]
            df_results = df_results[existing_final_cols]
            df_results.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"\nResults successfully saved to {output_csv_path}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}. Columns attempted: {final_ordered_columns}")
            print("DataFrame columns available:", df_results.columns if 'df_results' in locals() else "N/A")
            print("Attempting to save with all available columns from DataFrame without reordering...")
            try:
                pd.DataFrame(all_results_data).to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"Fallback save to {output_csv_path} successful (column order might vary).")
            except Exception as e_fallback:
                 print(f"Fallback CSV save also failed: {e_fallback}")

    # --- Generate Markdown Report ---
    md_content = f"# Analysis Results ({run_start_datetime.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
    md_content += f"- Input CSV: `{args.csv_file}`\n"
    md_content += f"- Processed Rows: {len(all_results_data)} (out of {total_rows_in_input_list} initially prepared items)\n" # Actual processed vs prepared
    md_content += f"- Total Wall Clock Processing Time for all items: {total_processing_duration:.2f} seconds\n"

    # Calculate and add average processing times from individual measurements
    if args.run_sentiment:
        sentiment_times = [r.get("Sentiment_Processing_Time_s") for r in all_results_data if r.get("Sentiment_Processing_Time_s") is not None]
        if sentiment_times:
            avg_sentiment_time = sum(sentiment_times) / len(sentiment_times)
            md_content += f"- Average Sentiment Processing Time per Row (measured): {avg_sentiment_time:.3f} seconds\n"
        else:
            md_content += f"- Average Sentiment Processing Time per Row (measured): N/A (no valid times recorded)\n"
            
    if args.run_topic:
        topic_times = [r.get("Topic_Processing_Time_s") for r in all_results_data if r.get("Topic_Processing_Time_s") is not None]
        if topic_times:
            avg_topic_time = sum(topic_times) / len(topic_times)
            md_content += f"- Average Topic Processing Time per Row (measured): {avg_topic_time:.3f} seconds\n"
        else:
            md_content += f"- Average Topic Processing Time per Row (measured): N/A (no valid times recorded)\n"

    if args.run_sentiment:
        md_content += f"- Sentiment Model: `{args.sentiment_model}`\n"
    if args.run_topic:
        md_content += f"- Topic Model: `{args.topic_model}`\n"
    md_content += f"- Output CSV: `{output_csv_path.name}`\n"
    md_content += f"- Output Markdown: `{output_md_path.name}`\n\n"

    if args.run_sentiment and all_results_data:
        sentiment_model_col_name = f"Sentiment ({args.sentiment_model.replace(':', '_')})"
        # Check if the column exists in the list of dicts (e.g., in the first dict)
        if all_results_data and sentiment_model_col_name in all_results_data[0]:
            metrics_summary = calculate_metrics_manually(all_results_data, sentiment_model_col_name, GROUND_TRUTH_COLUMN)
            if metrics_summary:
                md_content += f"## Sentiment Analysis Metrics for `{args.sentiment_model}`\n\n"
                md_content += f"- Accuracy: {metrics_summary['accuracy']:.4f} ({metrics_summary['correct_predictions']}/{metrics_summary['valid_predictions']} valid predictions)\n"
                for cls, cls_metrics in metrics_summary['class_metrics'].items():
                    md_content += f"  - {cls.upper()}: Support={cls_metrics['support']}, P={cls_metrics['precision']:.2f}, R={cls_metrics['recall']:.2f}, F1={cls_metrics['f1']:.2f}\n"
                md_content += "\n"
        else:
            md_content += f"## Sentiment Analysis Metrics for `{args.sentiment_model}`\n\n"
            md_content += f"- Metrics could not be calculated. Column '{sentiment_model_col_name}' likely not found in all results or no results processed for sentiment.\n\n"

    if args.run_topic:
        md_content += f"## Topic Modeling Results for `{args.topic_model}`\n\n"
        md_content += "Topic keywords generated per row. (No aggregate metrics for topics in this version).\n\n"

    md_content += "## Detailed Results Preview (Sample - first 20 rows)\n\n"
    table_header = [DOC_ID_COLUMN, GROUND_TRUTH_COLUMN] # Use constants
    if args.run_sentiment:
        sm_name_formatted = args.sentiment_model.replace(':', '_')
        table_header.append(f"Sentiment ({sm_name_formatted})")
        table_header.append(f"Sentiment ({sm_name_formatted}) Correct")
        table_header.append("Sentiment_Processing_Time_s")
    if args.run_topic:
        tm_name_formatted = args.topic_model.replace(':', '_')
        table_header.append(f"Topic Keyword ({tm_name_formatted})")
        table_header.append("Topic_Processing_Time_s")
    
    md_content += "|" + " | ".join(table_header) + "|\n"
    md_content += "|" + "---|" * len(table_header) + "\n"

    for i, res in enumerate(all_results_data[:20]):
        row_values = [str(res.get(DOC_ID_COLUMN, "N/A")), str(res.get(GROUND_TRUTH_COLUMN, "N/A"))]
        if args.run_sentiment:
            sm_name_formatted = args.sentiment_model.replace(':', '_')
            row_values.append(str(res.get(f"Sentiment ({sm_name_formatted})", "N/A")))
            row_values.append(str(res.get(f"Sentiment ({sm_name_formatted}) Correct", "N/A")))
            sent_time = res.get("Sentiment_Processing_Time_s")
            row_values.append(f"{sent_time:.3f}s" if sent_time is not None else "N/A")
        if args.run_topic:
            tm_name_formatted = args.topic_model.replace(':', '_')
            row_values.append(str(res.get(f"Topic Keyword ({tm_name_formatted})", "N/A")))
            top_time = res.get("Topic_Processing_Time_s")
            row_values.append(f"{top_time:.3f}s" if top_time is not None else "N/A")
        md_content += "|" + " | ".join(row_values) + "|\n"
    
    if len(all_results_data) > 20:
        md_content += f"| ... | (omitted {len(all_results_data) - 20} more rows) | ... | \n"

    try:
        with open(output_md_path, 'w', encoding='utf-8') as f_md:
            f_md.write(md_content)
        print(f"Markdown report successfully saved to {output_md_path}")
    except Exception as e:
        print(f"Error saving markdown report: {e}")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()