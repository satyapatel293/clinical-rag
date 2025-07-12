"""Clinical generation pipeline for RAG response generation."""

from typing import Dict, Any, List
from zenml import pipeline

from steps.format_retrieval_context import format_retrieval_context
from steps.build_clinical_prompt import build_clinical_prompt, build_simple_clinical_prompt
from steps.generate_with_ollama import generate_with_ollama, verify_ollama_model
from steps.parse_and_validate_response import parse_and_validate_response, validate_clinical_content
from steps.extract_metadata_and_citations import extract_metadata_and_citations, format_final_cli_output


# @pipeline(enable_cache=False)
def clinical_generation_pipeline(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    output_format: str = "detailed",
    include_full_metadata: bool = True
) -> Dict[str, Any]:
    """
    Complete pipeline for generating clinical responses with RAG.
    
    Args:
        query: User's clinical question
        retrieved_chunks: List of retrieved chunks from search
        model_name: Ollama model to use for generation
        temperature: Generation temperature (0.0-1.0)
        output_format: Output format for CLI ('detailed', 'simple', 'json')
        include_full_metadata: Whether to include detailed metadata
        
    Returns:
        Final formatted response ready for CLI display
    """
    # Step 1: Format retrieval context
    formatting_result = format_retrieval_context(
        query=query,
        retrieved_chunks=retrieved_chunks
    )
    
    # Step 2: Build clinical prompt
    prompt_result = build_clinical_prompt(
        formatting_result=formatting_result
    )
    
    # Step 3: Generate with Ollama LLM
    generation_result = generate_with_ollama(
        prompt_result=prompt_result,
        model_name=model_name,
        temperature=temperature
    )
    
    # Step 4: Parse and validate response
    parsing_result = parse_and_validate_response(
        generation_result=generation_result,
        require_structure=True
    )
    
    # Step 5: Extract metadata and citations
    final_result = extract_metadata_and_citations(
        parsing_result=parsing_result,
        formatting_result=formatting_result,
        include_full_metadata=include_full_metadata
    )
    
    # Step 6: Format for CLI output
    cli_result = format_final_cli_output(
        final_result=final_result,
        output_format=output_format
    )
    
    return cli_result


# @pipeline(enable_cache=False)
def simple_clinical_generation_pipeline(
    query: str,
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    output_format: str = "simple"
) -> Dict[str, Any]:
    """
    Simple pipeline for generating clinical responses without context.
    
    Args:
        query: Clinical question
        model_name: Ollama model to use
        temperature: Generation temperature
        output_format: Output format for CLI
        
    Returns:
        Simple clinical response
    """
    # Build simple prompt without context
    prompt_result = build_simple_clinical_prompt(
        query=query
    )
    
    # Generate with Ollama
    generation_result = generate_with_ollama(
        prompt_result=prompt_result,
        model_name=model_name,
        temperature=temperature
    )
    
    # Parse response (less strict validation)
    parsing_result = parse_and_validate_response(
        generation_result=generation_result,
        require_structure=False,
        min_length=20
    )
    
    # Create simple final result structure
    if parsing_result['success']:
        simple_result = {
            'final_response': {
                'response': parsing_result['parsed_response']['text'],
                'metadata': {
                    'generation_info': {
                        'model_used': generation_result['metadata']['model_name'],
                        'generation_time': generation_result['metadata']['generation_time_seconds'],
                        'temperature': temperature
                    },
                    'content_info': {
                        'word_count': parsing_result['parsed_response']['word_count'],
                        'quality_score': parsing_result['parsed_response']['quality_score']
                    }
                },
                'query': query,
                'citations': []  # No citations for simple generation
            },
            'success': True,
            'error': None
        }
    else:
        simple_result = {
            'final_response': {},
            'success': False,
            'error': parsing_result['error']
        }
    
    # Format for CLI
    cli_result = format_final_cli_output(
        final_result=simple_result,
        output_format=output_format
    )
    
    return cli_result


# @pipeline(enable_cache=False)
def clinical_generation_with_validation_pipeline(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    required_clinical_elements: List[str] = None
) -> Dict[str, Any]:
    """
    Enhanced pipeline with additional clinical validation.
    
    Args:
        query: User's clinical question
        retrieved_chunks: Retrieved chunks from search
        model_name: Ollama model to use
        temperature: Generation temperature
        required_clinical_elements: Required clinical elements for validation
        
    Returns:
        Clinical response with enhanced validation
    """
    # Run standard generation pipeline steps
    formatting_result = format_retrieval_context(
        query=query,
        retrieved_chunks=retrieved_chunks
    )
    
    prompt_result = build_clinical_prompt(
        formatting_result=formatting_result
    )
    
    generation_result = generate_with_ollama(
        prompt_result=prompt_result,
        model_name=model_name,
        temperature=temperature
    )
    
    parsing_result = parse_and_validate_response(
        generation_result=generation_result,
        require_structure=True
    )
    
    # Additional clinical validation step
    clinical_validation_result = validate_clinical_content(
        parsed_result=parsing_result,
        required_elements=required_clinical_elements
    )
    
    # Extract metadata with validation results
    final_result = extract_metadata_and_citations(
        parsing_result=parsing_result,
        formatting_result=formatting_result,
        include_full_metadata=True
    )
    
    # Add clinical validation to metadata
    if final_result['success'] and clinical_validation_result['success']:
        final_result['final_response']['detailed_metadata']['clinical_validation'] = clinical_validation_result['clinical_validation']
        final_result['final_response']['metadata']['clinical_requirements_met'] = clinical_validation_result['requirements_met']
    
    # Format for CLI with validation info
    cli_result = format_final_cli_output(
        final_result=final_result,
        output_format="detailed"
    )
    
    return cli_result
