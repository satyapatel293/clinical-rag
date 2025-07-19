"""ZenML pipeline for automated generation evaluation using LLM-as-a-Judge."""

from typing import Dict, Any, List
from zenml import pipeline

from steps.load_evaluation_dataset import load_evaluation_dataset
from steps.generate_and_evaluate_responses import generate_and_evaluate_responses
from steps.calculate_generation_metrics import calculate_generation_metrics
from steps.generate_generation_report import generate_generation_report


@pipeline(enable_cache=False)
def generation_evaluation_pipeline(
    dataset_path: str = "evaluation/test_datasets/clinical_qa_pairs.json",
    generation_model: str = "ollama/llama3.2:3b",
    judge_model: str = "openai/gpt-3.5-turbo",
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    enhance_query: bool = True,
    temperature: float = 0.1,
    save_detailed_report: bool = True,
    report_output_dir: str = "evaluation/reports/generation"
) -> Dict[str, Any]:
    """
    Complete automated pipeline for generation evaluation using LLM-as-a-Judge.
    
    This pipeline:
    1. Loads the evaluation dataset
    2. For each question, retrieves context and generates a response
    3. Uses LLM-as-a-Judge to evaluate each generated response
    4. Calculates aggregate metrics and generates reports
    
    Args:
        dataset_path: Path to evaluation dataset JSON file
        generation_model: Model to use for generating responses
        judge_model: Model to use for LLM-as-a-Judge evaluation
        top_k: Number of top results to retrieve for generation
        similarity_threshold: Minimum similarity threshold for retrieval
        enhance_query: Whether to enhance queries before retrieval
        temperature: Generation temperature for responses
        save_detailed_report: Whether to save detailed JSON reports to disk
        report_output_dir: Directory to save detailed reports
        
    Returns:
        Final evaluation report with metrics and analysis
    """
    
    # Step 1: Load evaluation dataset
    dataset_result = load_evaluation_dataset(
        dataset_path=dataset_path
    )
    
    # Step 2: Generate responses and evaluate with LLM judge
    evaluation_results = generate_and_evaluate_responses(
        dataset_result=dataset_result,
        generation_model=generation_model,
        judge_model=judge_model,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        enhance_query=enhance_query,
        temperature=temperature
    )
    
    # Step 3: Calculate aggregate metrics
    metrics_result = calculate_generation_metrics(
        evaluation_results=evaluation_results
    )
    
    # Step 4: Generate comprehensive report
    final_report = generate_generation_report(
        metrics_result=metrics_result,
        output_dir=report_output_dir,
        save_detailed=save_detailed_report
    )
    
    return final_report