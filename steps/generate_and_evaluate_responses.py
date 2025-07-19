"""Step to generate responses and evaluate them with LLM-as-a-Judge."""

from typing import Dict, Any, Annotated
from zenml import step
from zenml.logger import get_logger

from utils.search import ClinicalRAGSearcher
from steps.format_retrieval_context import format_retrieval_context
from steps.build_clinical_prompt import build_clinical_prompt
from steps.generate_with_litellm import generate_with_litellm
from steps.llm_judge_evaluation import llm_judge_evaluation

logger = get_logger(__name__)


@step
def generate_and_evaluate_responses(
    dataset_result: Dict[str, Any],
    generation_model: str = "ollama/llama3.2:3b",
    judge_model: str = "openai/gpt-3.5-turbo",
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    enhance_query: bool = True,
    temperature: float = 0.1
) -> Annotated[Dict[str, Any], "generation_evaluation_results"]:
    """
    Generate responses for each question and evaluate with LLM judge.
    
    Args:
        dataset_result: Result from load_evaluation_dataset step
        generation_model: Model for generating responses
        judge_model: Model for judging responses
        top_k: Number of chunks to retrieve
        similarity_threshold: Minimum similarity for retrieval
        enhance_query: Whether to enhance queries
        temperature: Generation temperature
        
    Returns:
        Dictionary containing all evaluation results
    """
    if not dataset_result['success']:
        return {
            'success': False,
            'error': f"Dataset loading failed: {dataset_result['error']}",
            'results': None,
            'parameters': None
        }
    
    try:
        # Initialize searcher for retrieval
        searcher = ClinicalRAGSearcher()
        
        evaluation_pairs = dataset_result['dataset_info']['evaluation_pairs']
        
        # Parameters for this evaluation run
        parameters = {
            'generation_model': generation_model,
            'judge_model': judge_model,
            'top_k': top_k,
            'similarity_threshold': similarity_threshold,
            'enhance_query': enhance_query,
            'temperature': temperature,
            'total_queries': len(evaluation_pairs)
        }
        
        # Process each question
        results = []
        for i, eval_pair in enumerate(evaluation_pairs, 1):
            question = eval_pair['question']
            
            logger.info(f"[{i}/{len(evaluation_pairs)}] Processing: {question[:60]}...")
            
            try:
                # Step 1: Retrieve relevant chunks
                retrieved_chunks = searcher.search_similar_chunks(
                    query=question,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    enhance_query=enhance_query
                )
                
                if not retrieved_chunks:
                    logger.warning(f"No chunks retrieved for question: {question[:60]}...")
                    results.append({
                        'id': eval_pair['id'],
                        'question': question,
                        'section_type': eval_pair['section_type'],
                        'filename': eval_pair['filename'],
                        'success': False,
                        'error': 'No chunks retrieved',
                        'judge_score': 0,
                        'judge_percentage': 0.0
                    })
                    continue
                
                # Step 2: Format context
                formatting_result = format_retrieval_context(
                    query=question,
                    retrieved_chunks=retrieved_chunks
                )
                
                if not formatting_result['success']:
                    results.append({
                        'id': eval_pair['id'],
                        'question': question,
                        'section_type': eval_pair['section_type'],
                        'filename': eval_pair['filename'],
                        'success': False,
                        'error': f"Context formatting failed: {formatting_result['error']}",
                        'judge_score': 0,
                        'judge_percentage': 0.0
                    })
                    continue
                
                # Step 3: Build prompt
                prompt_result = build_clinical_prompt(
                    formatting_result=formatting_result
                )
                
                if not prompt_result['success']:
                    results.append({
                        'id': eval_pair['id'],
                        'question': question,
                        'section_type': eval_pair['section_type'],
                        'filename': eval_pair['filename'],
                        'success': False,
                        'error': f"Prompt building failed: {prompt_result['error']}",
                        'judge_score': 0,
                        'judge_percentage': 0.0
                    })
                    continue
                
                # Step 4: Generate response
                generation_result = generate_with_litellm(
                    prompt_result=prompt_result,
                    model_name=generation_model,
                    temperature=temperature
                )
                
                if not generation_result['success']:
                    results.append({
                        'id': eval_pair['id'],
                        'question': question,
                        'section_type': eval_pair['section_type'],
                        'filename': eval_pair['filename'],
                        'success': False,
                        'error': f"Generation failed: {generation_result['error']}",
                        'judge_score': 0,
                        'judge_percentage': 0.0
                    })
                    continue
                
                generated_response = generation_result['generated_response']
                
                # Step 5: Format retrieved context for judge
                context_text = "\n\n".join([
                    f"Document: {chunk.get('filename', 'Unknown')}\n{chunk.get('text', '')}"
                    for chunk in retrieved_chunks
                ])
                
                # Step 6: Evaluate with LLM judge
                judge_result = llm_judge_evaluation(
                    question=question,
                    generated_response=generated_response,
                    retrieved_context=context_text,
                    judge_model=judge_model,
                    temperature=0.0  # Always use 0 temperature for consistent judging
                )
                
                # Step 7: Compile results
                result_data = {
                    'id': eval_pair['id'],
                    'question': question,
                    'section_type': eval_pair['section_type'],
                    'filename': eval_pair['filename'],
                    'generated_response': generated_response,
                    'retrieved_chunks_count': len(retrieved_chunks),
                    'success': judge_result['success'],
                    'judge_score': judge_result['final_score'],
                    'judge_percentage': judge_result['percentage'],
                    'individual_scores': judge_result['individual_scores'],
                    'dimension_scores': judge_result.get('dimension_scores', {}),
                    'parse_success': judge_result.get('parse_success', False)
                }
                
                if not judge_result['success']:
                    result_data['error'] = judge_result['error']
                
                logger.info(f"  Judge Score: {judge_result['final_score']}/7 ({judge_result['percentage']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to process question {i}: {e}")
                result_data = {
                    'id': eval_pair['id'],
                    'question': question,
                    'section_type': eval_pair.get('section_type', 'unknown'),
                    'filename': eval_pair.get('filename', 'unknown'),
                    'success': False,
                    'error': str(e),
                    'judge_score': 0,
                    'judge_percentage': 0.0
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
        logger.error(f"Generation evaluation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': None,
            'parameters': parameters if 'parameters' in locals() else None
        }