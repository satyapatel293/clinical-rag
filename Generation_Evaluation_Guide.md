# Clinical RAG Generation Evaluation Guide

## LLM-as-a-Judge Implementation

This system uses an LLM-as-a-Judge approach to evaluate the quality of generated clinical responses using binary questions and simple counting.

## How It Works

### Evaluation Method
- **Single Prompt**: One API call per evaluation (efficient and cost-effective)
- **7 Binary Questions**: Simple Yes/No questions across 3 clinical dimensions
- **Simple Scoring**: Count "Yes" answers for final score (X/7)

### Evaluation Dimensions

#### 1. Clinical Accuracy (3 questions)
1. Is the medical information factually correct?
2. Are any treatment recommendations clinically appropriate?
3. Would this information be safe for clinical decision-making?

#### 2. Relevance (2 questions)
4. Does this directly answer the clinical question asked?
5. Would this response help the healthcare provider with their query?

#### 3. Evidence Support (2 questions)
6. Is each medical claim supported by the retrieved documents?
7. Are the citations relevant and appropriate?

## Usage

### Basic Generation Evaluation
```bash
# Run with default settings (GPT-3.5-turbo as judge)
python run.py eval-generation

# Use GPT-4 as judge for higher accuracy
python run.py eval-generation --judge-model openai/gpt-4

# Test different generation models
python run.py eval-generation --generation-model anthropic/claude-3-haiku
```

### Advanced Options
```bash
# Custom retrieval parameters
python run.py eval-generation --top-k 10 --threshold 0.4

# Different temperature for generation
python run.py eval-generation --temperature 0.2

# Don't save detailed reports
python run.py eval-generation --no-save
```

## Output Interpretation

### Scoring Scale
- **X/7**: Number of "Yes" answers out of 7 questions
- **Percentage**: (X/7) * 100 for easy interpretation
- **Pass Rate**: Percentage scoring ≥5/7 (considered "passing")

### Performance Categories
- **Excellent (≥80%)**: Production-ready, high-quality generations
- **Good (70-79%)**: Strong performance, minor improvements needed
- **Fair (60-69%)**: Moderate performance, optimization required
- **Needs Improvement (<60%)**: Significant issues, major changes needed

### Sample Output
```
🧪 Clinical RAG - Generation Evaluation (LLM-as-a-Judge)
============================================================

🎯 Key Metrics:
  Overall Score: 4.2/7 (60.0%)
  Pass Rate: 65.4%
  Performance: FAIR

  Clinical Accuracy: 1.8/3 (60.0%)
  Relevance: 1.6/2 (80.0%)
  Evidence Support: 0.8/2 (40.0%)

💡 Recommendation: fair_needs_optimization
📝 Insights: Moderate performance, review generation prompts and training data
```

## Reports and Metadata

### ZenML Integration
- All evaluation results automatically saved as ZenML metadata
- Track performance across different models and configurations
- View results in ZenML dashboard

### File Outputs
- **Detailed JSON**: Complete evaluation results with individual scores
- **Summary TXT**: Human-readable summary report
- **Location**: `evaluation/reports/generation/`

### Example Report Files
```
evaluation/reports/generation/
├── generation_evaluation_20250119_143025.json
└── generation_summary_20250119_143025.txt
```

## Cost Optimization

### Model Selection
- **GPT-3.5-turbo**: 10x cheaper than GPT-4, good performance with examples
- **GPT-4**: Higher accuracy, more expensive
- **Local models**: Consider Ollama for cost-free evaluation

### Efficiency Features
- **Single prompt per evaluation**: Not 7 separate API calls
- **Batch processing**: Evaluates all 52 questions in one pipeline run
- **Smart parsing**: Robust extraction of scores from responses

## Best Practices

### For Reliable Results
1. **Consistent judge model**: Use same judge across comparisons
2. **Zero temperature**: Always use temperature=0.0 for judge consistency
3. **Multiple runs**: Run evaluations multiple times for statistical significance

### For Development
1. **Start with GPT-3.5**: Cheaper for initial testing
2. **Use GPT-4**: For final validation and production assessment
3. **Track over time**: Monitor performance changes with code updates

## Troubleshooting

### Common Issues
- **Parsing failures**: System automatically counts Yes/No if score extraction fails
- **No chunks retrieved**: Questions with no retrieval get 0/7 score
- **Generation failures**: Failed generations get 0/7 score
- **Judge API errors**: Retries and error logging built-in

### Environment Requirements
- **API Keys**: Ensure OPENAI_API_KEY or ANTHROPIC_API_KEY is set
- **LiteLLM**: Supports 100+ model providers
- **PostgreSQL**: Database must be running for retrieval

## Integration with Existing System

### Reuses Current Infrastructure
- **Same dataset**: Uses existing `clinical_qa_pairs.json` (52 questions)
- **Same retrieval**: Leverages current search and chunking system
- **Same generation**: Uses existing LiteLLM integration
- **ZenML pipelines**: Fits seamlessly into current architecture

### Complements Retrieval Evaluation
- **Retrieval eval**: Tests search quality (Hit Rate@K, MRR)
- **Generation eval**: Tests response quality (Clinical accuracy, relevance)
- **Together**: Complete RAG system assessment

This system provides reliable, cost-effective evaluation of clinical RAG generation quality using research-backed binary evaluation methods.