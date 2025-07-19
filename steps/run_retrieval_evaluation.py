"""Step to run retrieval evaluation against the clinical RAG system."""

from typing import Dict, Any, List, Annotated
from zenml import step

from utils.search import ClinicalRAGSearcher


@step
def run_retrieval_evaluation(
    dataset_result: Dict[str, Any],
    top_k: int = 10,
    similarity_threshold: float = 0.3,
    enhance_query: bool = True
) -> Annotated[Dict[str, Any], "evaluation_results"]:
    """
    Run retrieval evaluation using loaded dataset.
    
    Args:
        dataset_result: Result from load_evaluation_dataset step
        top_k: Number of top results to retrieve
        similarity_threshold: Minimum similarity threshold
        enhance_query: Whether to enhance queries
        
    Returns:
        Dictionary containing detailed evaluation results
    """
    if not dataset_result['success']:
        return {
            'success': False,
            'error': f"Dataset loading failed: {dataset_result['error']}",
            'results': None,
            'parameters': None
        }
    
    try:
        # Initialize searcher
        searcher = ClinicalRAGSearcher()
        
        evaluation_pairs = dataset_result['dataset_info']['evaluation_pairs']
        
        # Parameters for this evaluation run
        parameters = {
            'top_k': top_k,
            'similarity_threshold': similarity_threshold,
            'enhance_query': enhance_query,
            'total_queries': len(evaluation_pairs)
        }
        
        # Run evaluation for each query
        results = []
        for eval_pair in evaluation_pairs:
            question = eval_pair['question']
            expected_chunk_id = eval_pair['expected_chunk_id']
            
            try:
                # Search for similar chunks
                search_results = searcher.search_similar_chunks(
                    query=question,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    enhance_query=enhance_query
                )
                
                # Find the rank of the expected chunk
                found_at_rank = None
                for i, result in enumerate(search_results, 1):
                    if result.get('chunk_id') == expected_chunk_id:
                        found_at_rank = i
                        break
                
                result_data = {
                    'id': eval_pair['id'],
                    'question': question,
                    'expected_chunk_id': expected_chunk_id,
                    'found_at_rank': found_at_rank,
                    'num_results': len(search_results),
                    'section_type': eval_pair['section_type'],
                    'filename': eval_pair['filename'],
                    'success': found_at_rank is not None,
                    'top_similarity': search_results[0]['similarity'] if search_results else 0.0,
                    'retrieved_chunk_ids': [r.get('chunk_id') for r in search_results]
                }
                
            except Exception as e:
                result_data = {
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
            
            results.append(result_data)
        
        return {
            'success': True,
            'results': results,
            'parameters': parameters,
            'dataset_info': dataset_result['dataset_info'],
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'results': None,
            'parameters': parameters if 'parameters' in locals() else None
        }