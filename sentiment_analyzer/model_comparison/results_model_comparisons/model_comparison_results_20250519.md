# Sentiment Analysis Model Comparison (2025-05-19)

This document compares the performance of various LLM models for sentiment analysis, based on test results from the same dataset (32 samples).

## Summary of Results

| Model | Accuracy | Positive F1 | Neutral F1 | Negative F1 | Avg. Time/Row (s) | Total Time (s) |
|-------|----------|-------------|------------|-------------|-------------------|----------------|
| gemma3:27b-it-qat | 0.7812 | 0.67 | 0.80 | 0.86 | 1.852 | 15.30 |
| gemma3:4b-it-qat | 0.7500 | 0.59 | 0.75 | 0.87 | 0.444 | 3.66 |
| mixtral:latest | 0.7500 | 0.67 | 0.70 | 0.90 | 3.405 | 27.62 |
| llama3.3:70b | 0.7188 | 0.60 | 0.73 | 0.82 | 7.328 | 59.73 |
| gemma3:12b-it-qat | 0.6875 | 0.50 | 0.69 | 0.84 | 0.854 | 7.04 |
| llama3.2:3b | 0.5938 | 0.17 | 0.62 | 0.80 | 0.430 | 3.51 |

## Detailed Analysis

### Performance by Metric

- **Highest Overall Accuracy**: gemma3:27b-it-qat (78.12%)
- **Best Positive Sentiment Detection**: Tie between gemma3:27b-it-qat and mixtral:latest (F1 = 0.67)
- **Best Neutral Sentiment Detection**: gemma3:27b-it-qat (F1 = 0.80)
- **Best Negative Sentiment Detection**: mixtral:latest (F1 = 0.90)
- **Fastest Processing**: gemma3:4b-it-qat (0.444s per row)
- **Slowest Processing**: llama3.3:70b (7.328s per row)

### Per-Model Analysis

1. **gemma3:27b-it-qat**
   - Highest overall accuracy (78.12%)
   - Well-balanced performance across all sentiment classes
   - Good processing speed (1.852s per row)
   - Excellent at detecting neutral and negative sentiments

2. **gemma3:4b-it-qat**
   - Excellent balance of accuracy (75%) and speed (0.444s per row)
   - Very strong at detecting negative sentiments (F1 = 0.87)
   - Weaker at detecting positive sentiments (F1 = 0.59)
   - 4x faster than the 27b model with only a slight accuracy drop

3. **mixtral:latest**
   - Strong overall accuracy (75%)
   - Best model for detecting negative sentiments (F1 = 0.90)
   - Good at detecting positive sentiments (F1 = 0.67)
   - Relatively slow processing (3.405s per row)

4. **llama3.3:70b**
   - Good accuracy (71.88%)
   - Balanced sentiment detection capabilities
   - By far the slowest model (7.328s per row)
   - Total processing time nearly a minute (59.73s)

5. **gemma3:12b-it-qat**
   - Moderate accuracy (68.75%)
   - Good detection of negative sentiments (F1 = 0.84)
   - Poor detection of positive sentiments (F1 = 0.50)
   - Good processing speed (0.854s per row)

6. **llama3.2:3b**
   - Lowest accuracy (59.38%)
   - Very poor at detecting positive sentiments (F1 = 0.17)
   - Only strong at detecting negative sentiments (F1 = 0.80)
   - Fast processing speed (0.430s per row)

## Recommendation

Based on the comparison, **gemma3:27b-it-qat** is recommended as the best model for sentiment analysis due to:

1. Highest overall accuracy (78.12%)
2. Balanced performance across all sentiment classes
3. Reasonable processing speed (1.852s per row)
4. Total processing time of 15.30 seconds for the entire dataset

For use cases where processing speed is more critical than absolute accuracy, **gemma3:4b-it-qat** would be a strong alternative, offering:
- Nearly equivalent accuracy (75%)
- 4x faster processing (0.444s per row)
- Excellent performance on negative sentiments (F1 = 0.87)

The llama3.2:3b model should be avoided due to its poor performance in sentiment analysis, particularly for positive sentiment detection. 