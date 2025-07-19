#!/usr/bin/env python3
"""
Run retrieval evaluation using generated question-chunk pairs.
"""

import os
import json
import argparse
from typing import List, Dict, Any

# Set environment variable for ZenML on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from utils.search import ClinicalRAGSearcher
from metrics import calculate_all_metrics, print_metrics_summary, analyze_failures, print_failure_analysis


def load_evaluation_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def evaluate_single_query(searcher: ClinicalRAGSearcher, 
                         eval_pair: Dict[str, Any], 
                         top_k: int = 10,
                         similarity_threshold: float = 0.3,
                         enhance_query: bool = True) -> Dict[str, Any]:
    """
    Evaluate a single query against the retrieval system.
    
    Args:
        searcher: Clinical RAG searcher instance
        eval_pair: Evaluation pair containing question and expected chunk ID
        top_k: Number of top results to retrieve
        similarity_threshold: Minimum similarity threshold
        enhance_query: Whether to enhance the query
    
    Returns:
        Dictionary containing evaluation results
    """
    question = eval_pair['question']
    expected_chunk_id = eval_pair['expected_chunk_id']
    
    try:
        # Search for similar chunks
        results = searcher.search_similar_chunks(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enhance_query=enhance_query
        )
        
        # Find the rank of the expected chunk
        found_at_rank = None
        for i, result in enumerate(results, 1):
            if result.get('chunk_id') == expected_chunk_id:
                found_at_rank = i
                break
        
        return {
            'id': eval_pair['id'],
            'question': question,
            'expected_chunk_id': expected_chunk_id,
            'found_at_rank': found_at_rank,
            'num_results': len(results),
            'section_type': eval_pair['section_type'],
            'filename': eval_pair['filename'],
            'success': found_at_rank is not None,
            'top_similarity': results[0]['similarity'] if results else 0.0,
            'retrieved_chunk_ids': [r.get('chunk_id') for r in results]
        }
        
    except Exception as e:
        return {
            'id': eval_pair['id'],
            'question': question,
            'expected_chunk_id': expected_chunk_id,
            'found_at_rank': None,
            'num_results': 0,
            'section_type': eval_pair['section_type'],
            'filename': eval_pair['filename'],
            'success': False,
            'error': str(e),
            'retrieved_chunk_ids': []
        }


def run_evaluation(dataset_path: str = "evaluation/test_datasets/clinical_qa_pairs.json",
                  top_k: int = 10,
                  similarity_threshold: float = 0.3,
                  enhance_query: bool = True,
                  output_path: str = None) -> Dict[str, Any]:
    """
    Run complete retrieval evaluation.
    
    Args:
        dataset_path: Path to evaluation dataset JSON file
        top_k: Number of top results to retrieve
        similarity_threshold: Minimum similarity threshold
        enhance_query: Whether to enhance queries
        output_path: Path to save detailed results (optional)
    
    Returns:
        Dictionary containing evaluation results and metrics
    """
    print("🔍 Loading evaluation dataset...")
    eval_dataset = load_evaluation_dataset(dataset_path)
    print(f"✅ Loaded {len(eval_dataset)} evaluation pairs")
    
    print("\n🏥 Initializing Clinical RAG Searcher...")
    searcher = ClinicalRAGSearcher()
    
    print(f"\n🧪 Running evaluation with parameters:")
    print(f"  Top K: {top_k}")
    print(f"  Similarity threshold: {similarity_threshold}")
    print(f"  Query enhancement: {enhance_query}")
    print()
    
    # Run evaluation for each query
    results = []
    for i, eval_pair in enumerate(eval_dataset, 1):
        print(f"[{i}/{len(eval_dataset)}] Evaluating: {eval_pair['question'][:60]}...")
        
        result = evaluate_single_query(
            searcher=searcher,
            eval_pair=eval_pair,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enhance_query=enhance_query
        )
        
        results.append(result)
        
        # Show immediate feedback
        if result['success']:
            print(f"  ✅ Found at rank {result['found_at_rank']}")
        else:
            print(f"  ❌ Not found")
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_all_metrics(results, k_values=[1, 3, 5, 10])
    
    # Analyze failures
    failure_analysis = analyze_failures(results, top_n=5)
    
    # Prepare complete results
    evaluation_results = {
        'parameters': {
            'top_k': top_k,
            'similarity_threshold': similarity_threshold,
            'enhance_query': enhance_query,
            'total_queries': len(eval_dataset)
        },
        'metrics': metrics,
        'failure_analysis': failure_analysis,
        'detailed_results': results
    }
    
    # Save detailed results if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"📄 Detailed results saved to {output_path}")
    
    return evaluation_results


def main():
    """Main function for running evaluation."""
    parser = argparse.ArgumentParser(description='Run retrieval evaluation')
    parser.add_argument('--dataset', default='evaluation/test_datasets/clinical_qa_pairs.json',
                       help='Path to evaluation dataset')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top results to retrieve')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Similarity threshold')
    parser.add_argument('--no-enhance', action='store_true',
                       help='Disable query enhancement')
    parser.add_argument('--output', help='Path to save detailed results')
    
    args = parser.parse_args()
    
    print("🏥 Clinical RAG - Retrieval Evaluation")
    print("=" * 50)
    
    try:
        # Run evaluation
        results = run_evaluation(
            dataset_path=args.dataset,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            enhance_query=not args.no_enhance,
            output_path=args.output
        )
        
        # Print results
        print_metrics_summary(results['metrics'], results['parameters']['total_queries'])
        print_failure_analysis(results['failure_analysis'])
        
        print("\n✅ Evaluation completed successfully!")
        
        # Return success/failure based on hit rate
        hit_rate_5 = results['metrics'].get('hit_rate_at_5', 0.0)
        if hit_rate_5 > 0.7:
            print(f"🎉 Great performance! Hit Rate@5 = {hit_rate_5:.3f}")
            return 0
        elif hit_rate_5 > 0.5:
            print(f"🔄 Good performance, room for improvement. Hit Rate@5 = {hit_rate_5:.3f}")
            return 0
        else:
            print(f"⚠️  Performance needs improvement. Hit Rate@5 = {hit_rate_5:.3f}")
            return 1
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())