"""Step to calculate evaluation metrics from retrieval results."""

from typing import Dict, Any, List, Annotated
from zenml import step
import numpy as np


@step
def calculate_evaluation_metrics(
    evaluation_result: Dict[str, Any],
    k_values: List[int] = None
) -> Annotated[Dict[str, Any], "evaluation_metrics"]:
    """
    Calculate comprehensive evaluation metrics from retrieval results.
    
    Args:
        evaluation_result: Result from run_retrieval_evaluation step
        k_values: List of K values to calculate Hit Rate@K for
        
    Returns:
        Dictionary containing calculated metrics and analysis
    """
    if not evaluation_result['success']:
        return {
            'success': False,
            'error': f"Evaluation failed: {evaluation_result['error']}",
            'metrics': None
        }
    
    try:
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        results = evaluation_result['results']
        parameters = evaluation_result['parameters']
        
        # Calculate Hit Rate@K for different K values
        metrics = {}
        for k in k_values:
            hits = sum(1 for r in results if r.get('found_at_rank') is not None and r['found_at_rank'] <= k)
            metrics[f'hit_rate_at_{k}'] = hits / len(results) if results else 0.0
        
        # Precision@1 (same as Hit Rate@1)
        metrics['precision_at_1'] = metrics.get('hit_rate_at_1', 0.0)
        
        # Mean Reciprocal Rank
        reciprocal_ranks = []
        for result in results:
            found_at_rank = result.get('found_at_rank')
            if found_at_rank is not None:
                reciprocal_ranks.append(1.0 / found_at_rank)
            else:
                reciprocal_ranks.append(0.0)
        metrics['mean_reciprocal_rank'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        # Additional statistics
        found_ranks = [r.get('found_at_rank') for r in results if r.get('found_at_rank') is not None]
        
        if found_ranks:
            metrics['average_rank'] = np.mean(found_ranks)
            metrics['median_rank'] = np.median(found_ranks)
            metrics['min_rank'] = min(found_ranks)
            metrics['max_rank'] = max(found_ranks)
        else:
            metrics['average_rank'] = None
            metrics['median_rank'] = None
            metrics['min_rank'] = None
            metrics['max_rank'] = None
        
        # Success rate (percentage of queries that found the target chunk at any rank)
        metrics['success_rate'] = len(found_ranks) / len(results) if results else 0.0
        
        # Failure analysis
        failures = [r for r in results if r.get('found_at_rank') is None]
        
        # Group failures by section type
        section_failures = {}
        for failure in failures:
            section = failure.get('section_type', 'unknown')
            section_failures[section] = section_failures.get(section, 0) + 1
        
        # Group failures by document
        doc_failures = {}
        for failure in failures:
            doc = failure.get('filename', 'unknown')
            doc_failures[doc] = doc_failures.get(doc, 0) + 1
        
        failure_analysis = {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(results) if results else 0.0,
            'failures_by_section': section_failures,
            'failures_by_document': doc_failures,
            'example_failures': failures[:5]  # Top 5 examples
        }
        
        # Performance categorization
        hit_rate_5 = metrics.get('hit_rate_at_5', 0.0)
        if hit_rate_5 >= 0.7:
            performance_category = "excellent"
        elif hit_rate_5 >= 0.5:
            performance_category = "good"
        elif hit_rate_5 >= 0.3:
            performance_category = "fair"
        else:
            performance_category = "needs_improvement"
        
        return {
            'success': True,
            'metrics': metrics,
            'failure_analysis': failure_analysis,
            'performance_category': performance_category,
            'parameters': parameters,
            'total_queries': len(results),
            'k_values': k_values,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'metrics': None
        }