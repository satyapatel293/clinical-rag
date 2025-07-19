"""
Evaluation metrics for retrieval performance.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def calculate_hit_rate_at_k(results: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Calculate Hit Rate@K - percentage of queries where target chunk appears in top K results.
    
    Args:
        results: List of evaluation results, each containing 'found_at_rank' or None
        k: Number of top results to consider
    
    Returns:
        Hit rate as a float between 0 and 1
    """
    if not results:
        return 0.0
    
    hits = 0
    for result in results:
        found_at_rank = result.get('found_at_rank')
        if found_at_rank is not None and found_at_rank <= k:
            hits += 1
    
    return hits / len(results)


def calculate_precision_at_1(results: List[Dict[str, Any]]) -> float:
    """
    Calculate Precision@1 - percentage of queries where target chunk is rank 1.
    
    Args:
        results: List of evaluation results, each containing 'found_at_rank' or None
    
    Returns:
        Precision@1 as a float between 0 and 1
    """
    if not results:
        return 0.0
    
    hits_at_1 = 0
    for result in results:
        found_at_rank = result.get('found_at_rank')
        if found_at_rank == 1:
            hits_at_1 += 1
    
    return hits_at_1 / len(results)


def calculate_mean_reciprocal_rank(results: List[Dict[str, Any]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) - average of reciprocal ranks of first relevant results.
    
    Args:
        results: List of evaluation results, each containing 'found_at_rank' or None
    
    Returns:
        MRR as a float between 0 and 1
    """
    if not results:
        return 0.0
    
    reciprocal_ranks = []
    for result in results:
        found_at_rank = result.get('found_at_rank')
        if found_at_rank is not None:
            reciprocal_ranks.append(1.0 / found_at_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def calculate_all_metrics(results: List[Dict[str, Any]], k_values: List[int] = None) -> Dict[str, Any]:
    """
    Calculate all evaluation metrics for a set of results.
    
    Args:
        results: List of evaluation results
        k_values: List of K values to calculate Hit Rate@K for (default: [1, 3, 5, 10])
    
    Returns:
        Dictionary containing all calculated metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    metrics = {}
    
    # Hit Rate@K for different K values
    for k in k_values:
        metrics[f'hit_rate_at_{k}'] = calculate_hit_rate_at_k(results, k)
    
    # Precision@1 (same as Hit Rate@1, but kept separate for clarity)
    metrics['precision_at_1'] = calculate_precision_at_1(results)
    
    # Mean Reciprocal Rank
    metrics['mean_reciprocal_rank'] = calculate_mean_reciprocal_rank(results)
    
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
    
    return metrics


def print_metrics_summary(metrics: Dict[str, Any], total_queries: int):
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary containing calculated metrics
        total_queries: Total number of queries evaluated
    """
    print("\n" + "=" * 50)
    print("📊 RETRIEVAL EVALUATION METRICS")
    print("=" * 50)
    
    print(f"Total queries: {total_queries}")
    print(f"Success rate: {metrics['success_rate']:.3f} ({metrics['success_rate']*100:.1f}%)")
    print()
    
    # Hit Rate@K
    print("🎯 Hit Rate@K (Target found in top K results):")
    for k in [1, 3, 5, 10]:
        if f'hit_rate_at_{k}' in metrics:
            rate = metrics[f'hit_rate_at_{k}']
            print(f"  Hit Rate@{k}: {rate:.3f} ({rate*100:.1f}%)")
    print()
    
    # Precision@1
    print(f"🥇 Precision@1: {metrics['precision_at_1']:.3f} ({metrics['precision_at_1']*100:.1f}%)")
    print()
    
    # Mean Reciprocal Rank
    print(f"📈 Mean Reciprocal Rank: {metrics['mean_reciprocal_rank']:.3f}")
    print()
    
    # Rank statistics
    if metrics['average_rank'] is not None:
        print("📊 Rank Statistics (when found):")
        print(f"  Average rank: {metrics['average_rank']:.1f}")
        print(f"  Median rank: {metrics['median_rank']:.1f}")
        print(f"  Best rank: {metrics['min_rank']}")
        print(f"  Worst rank: {metrics['max_rank']}")
    else:
        print("📊 No successful retrievals found")


def analyze_failures(results: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    """
    Analyze failed queries to identify patterns.
    
    Args:
        results: List of evaluation results
        top_n: Number of top failure examples to include
    
    Returns:
        Dictionary containing failure analysis
    """
    failures = [r for r in results if r.get('found_at_rank') is None]
    
    if not failures:
        return {'total_failures': 0, 'failure_rate': 0.0}
    
    # Group by section type
    section_failures = {}
    for failure in failures:
        section = failure.get('section_type', 'unknown')
        section_failures[section] = section_failures.get(section, 0) + 1
    
    # Group by document
    doc_failures = {}
    for failure in failures:
        doc = failure.get('filename', 'unknown')
        doc_failures[doc] = doc_failures.get(doc, 0) + 1
    
    analysis = {
        'total_failures': len(failures),
        'failure_rate': len(failures) / len(results) if results else 0.0,
        'failures_by_section': section_failures,
        'failures_by_document': doc_failures,
        'example_failures': failures[:top_n]
    }
    
    return analysis


def print_failure_analysis(analysis: Dict[str, Any]):
    """
    Print formatted failure analysis.
    
    Args:
        analysis: Dictionary containing failure analysis from analyze_failures()
    """
    if analysis['total_failures'] == 0:
        print("🎉 No failures found!")
        return
    
    print("\n" + "=" * 50)
    print("🔍 FAILURE ANALYSIS")
    print("=" * 50)
    
    print(f"Total failures: {analysis['total_failures']}")
    print(f"Failure rate: {analysis['failure_rate']:.3f} ({analysis['failure_rate']*100:.1f}%)")
    print()
    
    # Failures by section
    if analysis['failures_by_section']:
        print("📋 Failures by Section Type:")
        for section, count in sorted(analysis['failures_by_section'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {section}: {count}")
        print()
    
    # Failures by document
    if analysis['failures_by_document']:
        print("📄 Failures by Document:")
        for doc, count in sorted(analysis['failures_by_document'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc}: {count}")
        print()
    
    # Example failures
    if analysis['example_failures']:
        print("❌ Example Failed Queries:")
        for i, failure in enumerate(analysis['example_failures'], 1):
            print(f"  {i}. {failure.get('question', 'N/A')}")
            print(f"     Section: {failure.get('section_type', 'N/A')}")
            print(f"     Document: {failure.get('filename', 'N/A')}")
            print()