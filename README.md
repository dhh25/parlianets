# Parlament Analysis Tools

This directory contains tools for sentiment analysis and topic modeling of parliamentary speech data, specifically designed for analyzing Finnish parliament (Eduskunta) speeches.

## Components

- **sentiment_analyzer/**: Main analysis tools
  - **model_comparison/**: Tools for evaluating different LLM models
  - **parlament_analysis/**: Finnish parliament specific analysis
  - **raw_data/**: Input data files
  - **visualize_model_performance.py**: Visualization utility

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have Ollama installed and running on your system, as these tools leverage Ollama for accessing local LLMs.

## Quick Start

1. Ensure Ollama is running in the background:
```bash
ollama serve
```

2. For model comparison on test data:
```bash
python sentiment_analyzer/model_comparison/run_sentiment_analysis.py --run_sentiment --sentiment_model gemma3:4b-it-qat
```

3. For Finnish parliament data analysis:
```bash
python sentiment_analyzer/parlament_analysis/run_analysis_finnish_generic.py --run_sentiment --sentiment_model gemma3:4b-it-qat
```

4. To visualize results after running analyses:
```bash
python sentiment_analyzer/visualize_model_performance.py
```

## Advanced Usage

Both analysis scripts support multiple options:

- `--run_sentiment`: Run sentiment analysis (positive/negative/neutral)
- `--run_topic`: Run topic modeling (generate keywords)
- `--sentiment_model MODEL`: Specify which Ollama model to use for sentiment
- `--topic_model MODEL`: Specify which Ollama model to use for topics
- `--csv_file PATH`: Path to input CSV file
- `--output_dir PATH`: Directory to save results
- `--rows N`: Process only N rows (0 for all)
- `--max_workers N`: Number of concurrent API calls

## Notes on Finnish Parliament Dataset

The Finnish parliament dataset contains debate transcripts with each row containing:
- Speaker position and ID
- Speech text
- Context (surrounding speeches)

Results are saved in both CSV and Markdown format in their respective output folders.

## Tested Models and Results

The following Ollama models have been tested for sentiment analysis and topic modeling:

### Sentiment Analysis Results

| Model | Size | Performance | Notes |
|-------|------|-------------|-------|
| gemma3:4b-it-qat | 4B | Good baseline performance | Fast, balanced accuracy/speed |
| gemma3:12b-it-qat | 12B | Better accuracy than 4B | 2-3x slower than 4B |
| gemma3:27b-it-qat | 27B | Best accuracy in Gemma family | 4-5x slower than 4B |
| llama3.1:8b | 8B | Competitive with Gemma 12B | Good multilingual capability |
| llama3.2:3b | 3B | Surprisingly strong for size | Fast inference |
| llama3.3:70b | 70B | Highest accuracy overall | Much slower, high memory requirements |
| mixtral:latest | Mixture | Strong performance | Good multilingual capabilities |
| deepseek-r1:32b-8k | 32B | Competitive with largest models | Good for complex contexts |

### Topic Modeling Results

Topic modeling with the same models showed similar patterns, with larger models generally providing more relevant and nuanced keywords. The topic extraction is most effective when combined with sentiment analysis to provide context for the sentiment classifications.

### System Requirements

- Most models up to 12B parameters work well on modern laptops with 16GB RAM
- Larger models (27B+) benefit greatly from GPU acceleration
- The system automatically detects and optimizes for:
  - Apple Silicon (using MPS)
  - NVIDIA GPUs (using CUDA)
  - CPU-only environments

### Cross-platform Support

The tool works across operating systems:
- macOS (with optimizations for Apple Silicon)
- Linux (with NVIDIA GPU support where available)
- Windows (basic support)

## Hardware Detection and Optimization

The analysis scripts automatically detect your system hardware and configure optimizations accordingly:

### Automatic Environment Detection

1. **Apple Silicon (M-series)** 
   - Automatically detected on macOS with ARM processors
   - Utilizes Metal Performance Shaders (MPS) for GPU acceleration
   - Optimized parameters are applied: num_ctx=8192, num_gpu=99 (uses all available GPU memory)

2. **NVIDIA GPUs**
   - Automatically detected on Linux systems by checking for `nvidia-smi` command
   - Utilizes CUDA for GPU acceleration
   - If detected, applies optimized parameters: num_ctx=8192, num_gpu=1

3. **CPU-only Systems**
   - Falls back to CPU processing with optimized thread settings

### NVIDIA GPU Setup Guide

If you're using Linux with an NVIDIA GPU, follow these steps for optimal performance:

1. **Install NVIDIA Drivers**: 
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-XXX  # Replace XXX with latest version (e.g., 545)
   ```

2. **Install CUDA Toolkit**:
   ```bash
   # Download CUDA installer from NVIDIA website
   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
   sudo sh cuda_12.4.0_550.54.14_linux.run
   ```

3. **Install PyTorch with CUDA support**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Add CUDA to your PATH**:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

5. **Verify Installation**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

For macOS with Apple Silicon, no additional setup is required as the scripts automatically detect and use MPS acceleration.

---

## Suomeksi

Tämä kansio sisältää työkaluja eduskunnan puhedatan sentimenttianalyysiin ja aihemallinnukseen. Analyysi suoritetaan paikallisten kielimallien avulla Ollama-palvelua käyttäen.

### Käyttöönotto
1. Asenna riippuvuudet: `pip install -r requirements.txt`
2. Asenna Ollama ja varmista että se on käynnissä: `ollama serve`

### Peruskäyttö
- Sentimenttianalyysi suomenkieliselle datalle: `python sentiment_analyzer/parlament_analysis/run_analysis_finnish_generic.py --run_sentiment --sentiment_model gemma3:4b-it-qat`
- Tulosten visualisointi: `python sentiment_analyzer/visualize_model_performance.py` 

## Author

**Jussi Wright**  
Contact: jussi.wright@helsinki.fi

Copyright © 2025 