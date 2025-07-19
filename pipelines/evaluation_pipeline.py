"""ZenML pipeline for automated retrieval evaluation."""

from typing import Dict, Any, List
from zenml import pipeline

from steps.load_evaluation_dataset import load_evaluation_dataset
from steps.run_retrieval_evaluation import run_retrieval_evaluation
from steps.calculate_evaluation_metrics import calculate_evaluation_metrics
from steps.generate_evaluation_report import generate_evaluation_report


@pipeline(enable_cache=False)
def retrieval_evaluation_pipeline(
    dataset_path: str = "evaluation/test_datasets/clinical_qa_pairs.json",
    top_k: int = 10,
    similarity_threshold: float = 0.3,
    enhance_query: bool = True,
    k_values: List[int] = None,
    save_detailed_report: bool = True,
    report_output_dir: str = "evaluation/reports"
) -> Dict[str, Any]:
    """
    Complete automated pipeline for retrieval evaluation.
    
    This pipeline:
    1. Loads the evaluation dataset
    2. Runs retrieval evaluation against the clinical RAG system
    3. Calculates comprehensive metrics (Hit Rate@K, MRR, Precision@1)
    4. Generates detailed reports and saves as ZenML metadata
    
    Args:
        dataset_path: Path to evaluation dataset JSON file
        top_k: Number of top results to retrieve for evaluation
        similarity_threshold: Minimum similarity threshold for retrieval
        enhance_query: Whether to enhance queries before retrieval
        k_values: List of K values for Hit Rate@K calculation (default: [1,3,5,10])
        save_detailed_report: Whether to save detailed JSON reports to disk
        report_output_dir: Directory to save detailed reports
        
    Returns:
        Final evaluation report with metrics and analysis
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    # Step 1: Load evaluation dataset
    dataset_result = load_evaluation_dataset(
        dataset_path=dataset_path
    )
    
    # Step 2: Run retrieval evaluation
    evaluation_result = run_retrieval_evaluation(
        dataset_result=dataset_result,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        enhance_query=enhance_query
    )
    
    # Step 3: Calculate evaluation metrics
    metrics_result = calculate_evaluation_metrics(
        evaluation_result=evaluation_result,
        k_values=k_values
    )
    
    # Step 4: Generate comprehensive report and save as ZenML metadata
    final_report = generate_evaluation_report(
        metrics_result=metrics_result,
        output_dir=report_output_dir,
        save_detailed=save_detailed_report
    )
    
    return final_report


@pipeline(enable_cache=False)  
def comparative_evaluation_pipeline(
    dataset_path: str = "evaluation/test_datasets/clinical_qa_pairs.json",
    configurations: List[Dict[str, Any]] = None,
    k_values: List[int] = None
) -> Dict[str, Any]:
    """
    Pipeline to run comparative evaluation across multiple configurations.
    Note: This is a simplified approach - each configuration should ideally be its own pipeline run.
    """
    # For now, just run the baseline configuration and return that
    # In practice, you'd run multiple separate pipeline instances
    
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    # Run single evaluation with baseline configuration
    dataset_result = load_evaluation_dataset(dataset_path=dataset_path)
    
    evaluation_result = run_retrieval_evaluation(
        dataset_result=dataset_result,
        top_k=5,
        similarity_threshold=0.3,
        enhance_query=True
    )
    
    metrics_result = calculate_evaluation_metrics(
        evaluation_result=evaluation_result,
        k_values=k_values
    )
    
    # Return simplified comparative result
    comparative_summary = {
        'success': True,
        'note': 'Comparative evaluation requires running separate pipeline instances',
        'baseline_result': metrics_result
    }
    
    return comparative_summary