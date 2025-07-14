"""Clinical generation pipeline with enhanced validation."""

from typing import Dict, Any, List
from zenml import pipeline

from steps.format_retrieval_context import format_retrieval_context
from steps.build_clinical_prompt import build_clinical_prompt
from steps.generate_with_litellm import generate_with_litellm
from steps.parse_and_validate_response import parse_and_validate_response, validate_clinical_content
from steps.extract_metadata_and_citations import extract_metadata_and_citations, format_final_cli_output


@pipeline(enable_cache=False)
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
    
    generation_result = generate_with_litellm(
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