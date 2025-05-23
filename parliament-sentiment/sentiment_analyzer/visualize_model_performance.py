#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Visualization Tool

This script visualizes the performance of different language models used for sentiment analysis
and topic modeling. It reads the results from Markdown files generated by the analysis scripts,
extracts performance metrics (accuracy, processing time), and creates scatter plots comparing
these metrics across different models. The visualization helps identify the best models for
specific use cases based on the trade-off between accuracy and processing speed.

Author: Jussi Wright
Email: jussi.wright@helsinki.fi
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime

def extract_info_from_md(md_file_content_lines):
    """Extracts model name, accuracy, and average processing times from MD content."""
    sentiment_model_name = None
    topic_model_name = None
    accuracy = None
    avg_sentiment_time_s = None
    avg_topic_time_s = None
    model_size_b = None # For x-axis or labeling
    
    # Fix possible \n characters in content
    fixed_lines = []
    for line in md_file_content_lines:
        if "\n" in line:
            split_lines = line.split("\n")
            fixed_lines.extend([l.strip() for l in split_lines if l.strip()])
        else:
            fixed_lines.append(line.strip())
    
    for line in fixed_lines:
        if line.startswith("- Sentiment Model:"):
            match = re.search(r'`([^`]+)`', line)
            if match:
                sentiment_model_name = match.group(1).strip()
        elif line.startswith("- Topic Model:"):
            match = re.search(r'`([^`]+)`', line)
            if match:
                topic_model_name = match.group(1).strip()
        elif line.startswith("- Accuracy:"):
            try:
                match = re.search(r'Accuracy: ([\d\.]+)', line)
                if match:
                    accuracy = float(match.group(1))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse accuracy from line: {line} (Error: {e})")
        elif line.startswith("- Average Sentiment Processing Time per Row (measured):"):
            try:
                match = re.search(r'measured\): ([\d\.]+)', line)
                if match:
                    avg_sentiment_time_s = float(match.group(1))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse sentiment time from line: {line} (Error: {e})")
        elif line.startswith("- Average Topic Processing Time per Row (measured):"):
            try:
                match = re.search(r'measured\): ([\d\.]+)', line)
                if match:
                    avg_topic_time_s = float(match.group(1))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse topic time from line: {line} (Error: {e})")

    # Construct a display name and attempt to parse model size
    display_name = "Unknown Model"
    primary_model_for_size = None
    if sentiment_model_name and topic_model_name and sentiment_model_name == topic_model_name:
        display_name = sentiment_model_name
        primary_model_for_size = sentiment_model_name
    elif sentiment_model_name and topic_model_name:
        display_name = f"S: {sentiment_model_name} / T: {topic_model_name}"
        primary_model_for_size = sentiment_model_name # Default to sentiment model for size extraction
    elif sentiment_model_name:
        display_name = sentiment_model_name
        primary_model_for_size = sentiment_model_name
    elif topic_model_name:
        display_name = f"Topic Only: {topic_model_name}" # Should ideally not happen if we plot vs sentiment accuracy
        primary_model_for_size = topic_model_name
    
    if primary_model_for_size:
        match_size = re.search(r'(\d+\.?\d*)b', primary_model_for_size, re.IGNORECASE)
        if match_size:
            try:
                model_size_b = float(match_size.group(1))
            except ValueError:
                pass # Keep model_size_b as None

    total_avg_time_s = 0
    has_time = False
    if avg_sentiment_time_s is not None:
        total_avg_time_s += avg_sentiment_time_s
        has_time = True
    if avg_topic_time_s is not None:
        total_avg_time_s += avg_topic_time_s
        has_time = True
    
    if not has_time:
        total_avg_time_s = None # If neither time was found/valid
    
    # Debug output for troubleshooting
    print(f"    Extracted: Name={display_name}, Size={model_size_b}, Accuracy={accuracy}, AvgTime={total_avg_time_s}")
        
    return display_name, model_size_b, accuracy, total_avg_time_s

def main():
    """Main function to process MD files and generate plot."""
    # Check both results directories
    results_dirs = ["Parlament/sentiment_analyzer/model_comparison/results_model_comparisons", "Parlament/sentiment_analyzer/parlament_analysis/results_finnish"]
    all_md_files = []
    
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            md_files = glob.glob(os.path.join(results_dir, "*.md"))
            if md_files:
                all_md_files.extend(md_files)
                print(f"Found {len(md_files)} MD files in {results_dir}/")

    if not all_md_files:
        print(f"No *.md files found in directories: {', '.join(results_dirs)}")
        print("Please ensure that:")
        print(f"1. You are running this script from the root directory of the project.")
        print(f"2. The mentioned results directories exist and contain '.md' files from the analysis scripts.")
        return

    plot_data = []

    print("Processing MD files...")
    for md_file_path in all_md_files:
        print(f"  - Reading: {md_file_path}")
        md_lines = None
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f_md:
                md_lines = [line for line in f_md.readlines()]
        except Exception as e:
            print(f"    Error reading MD file {md_file_path}: {e}")
            continue 

        if md_lines:
            name, size, acc, avg_time = extract_info_from_md(md_lines)
            
            # We need accuracy and average time for plotting against each other.
            # Model name is for labeling. Size is extra info.
            if name != "Unknown Model" and acc is not None and avg_time is not None:
                plot_data.append({
                    "model_name_display": name, 
                    "model_size_b": size, 
                    "accuracy": acc, 
                    "avg_total_time_per_row_s": avg_time,
                    "source_file": os.path.basename(md_file_path)
                })
            else:
                print(f"    Could not extract all required data from {md_file_path}. Name: {name}, Accuracy: {acc}, AvgTime: {avg_time}")

    if not plot_data:
        print("No plottable data extracted from MD files. Cannot generate plot.")
        return

    df = pd.DataFrame(plot_data)
    print("\n--- Extracted Data for Plotting ---")
    print(df.to_string())

    # Filter out rows where avg_total_time_per_row_s or accuracy is None (should be caught by 'if' above, but defensive)
    df_plottable = df.dropna(subset=['avg_total_time_per_row_s', 'accuracy'])

    if df_plottable.empty:
        print("\nNo plottable data after filtering. Check warnings above.")
        return

    # Sort by average time for potentially better visual connection if lines were drawn, or just for consistent ordering
    df_plottable = df_plottable.sort_values(by="avg_total_time_per_row_s")

    plt.figure(figsize=(14, 9))
    
    for index, row in df_plottable.iterrows():
        label_text = f"{row['model_name_display']}"
        if pd.notna(row['model_size_b']):
            label_text += f" ({row['model_size_b']}b)"
        
        plt.scatter(row['avg_total_time_per_row_s'], row['accuracy'], s=100, alpha=0.7) 
        plt.text(row['avg_total_time_per_row_s'] * 1.01, # Offset text slightly from point
                 row['accuracy'] * 1.01,
                 label_text, 
                 fontsize=8)

    plt.title('Model Performance: Accuracy vs. Avg. Measured Processing Time per Row')
    plt.xlabel('Avg. Total Measured Processing Time per Row (seconds) [Sentiment + Topic]')
    plt.ylabel('Sentiment Accuracy')
    plt.grid(True, which="both", ls="--", c='0.7')
    
    # Add a note if times are sum of sentiment and topic
    plt.figtext(0.5, 0.01, "Note: Processing time is the sum of measured sentiment and topic modeling times per row from MD reports.", 
                ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":3})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for figtext and title

    plot_filename = "model_performance_accuracy_vs_measured_time.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved as {plot_filename}")
    
if __name__ == "__main__":
    main() 