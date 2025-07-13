"""Simple clinical generation pipeline without context."""

from typing import Dict, Any
from zenml import pipeline

from steps.build_clinical_prompt import build_simple_clinical_prompt
from steps.generate_with_ollama import generate_with_ollama
from steps.parse_and_validate_response import parse_and_validate_response
from steps.extract_metadata_and_citations import format_final_cli_output


@pipeline(enable_cache=False)
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
    
    # Format for CLI output directly - let the step handle the structure
    cli_result = format_final_cli_output(
        final_result=parsing_result,
        output_format=output_format
    )
    
    return cli_result