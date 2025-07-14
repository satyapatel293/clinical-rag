"""Clinical generation pipeline for RAG response generation."""

from typing import Dict, Any, List
from zenml import pipeline

from steps.format_retrieval_context import format_retrieval_context
from steps.build_clinical_prompt import build_clinical_prompt
from steps.generate_with_litellm import generate_with_litellm
from steps.parse_and_validate_response import parse_and_validate_response
from steps.extract_metadata_and_citations import extract_metadata_and_citations, format_final_cli_output


@pipeline(enable_cache=False)
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
    generation_result = generate_with_litellm(
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



