"""Extract metadata and citations step for Clinical RAG generation pipeline."""

from typing import Dict, Any, List
from zenml import step


#@step
def extract_metadata_and_citations(
    parsing_result: Dict[str, Any],
    formatting_result: Dict[str, Any],
    include_full_metadata: bool = True
) -> Dict[str, Any]:
    """
    Extract final metadata and citations for the clinical response.
    
    Args:
        parsing_result: Output from parse_and_validate_response step
        formatting_result: Output from format_retrieval_context step (for citations)
        include_full_metadata: Whether to include detailed metadata
        
    Returns:
        Dictionary containing final response with citations and metadata
    """
    try:
        if not parsing_result['success']:
            return {
                'final_response': {},
                'success': False,
                'error': f"Cannot extract metadata - parsing failed: {parsing_result['error']}"
            }
        
        if not formatting_result['success']:
            return {
                'final_response': {},
                'success': False,
                'error': f"Cannot extract citations - formatting failed: {formatting_result['error']}"
            }
        
        parsed_response = parsing_result['parsed_response']
        formatted_context = formatting_result['formatted_context']
        
        # Extract citation information from the original chunks
        citations = []
        for chunk in formatted_context['chunks']:
            citation = {
                'source_id': chunk['source_id'],
                'document': chunk['document_name'],
                'section': chunk['section_type'],
                'relevance': chunk['similarity'],
                'excerpt': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            }
            citations.append(citation)
        
        # Sort citations by relevance (highest first)
        citations.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Create summary metadata
        response_metadata = {
            'generation_info': {
                'model_used': parsing_result['metadata']['generation_metadata'].get('model_name', 'unknown'),
                'generation_time': parsing_result['metadata']['generation_metadata'].get('generation_time_seconds', 0),
                'temperature': parsing_result['metadata']['generation_metadata'].get('temperature', 0.1),
                'total_tokens': parsing_result['metadata']['generation_metadata'].get('total_tokens', 0)
            },
            'content_info': {
                'response_length': parsed_response['character_count'],
                'word_count': parsed_response['word_count'],
                'quality_score': parsed_response['quality_score'],
                'validation_passed': parsed_response['validation_passed']
            },
            'source_info': {
                'chunks_used': len(citations),
                'documents_referenced': len(set(c['document'] for c in citations)),
                'sections_covered': list(set(c['section'] for c in citations)),
                'avg_relevance': sum(c['relevance'] for c in citations) / len(citations) if citations else 0.0,
                'highest_relevance': max(c['relevance'] for c in citations) if citations else 0.0
            }
        }
        
        # Build final response structure
        final_response = {
            'response': parsed_response['text'],
            'sections': parsed_response['sections'],
            'citations': citations,
            'metadata': response_metadata,
            'query': formatted_context['query']
        }
        
        # Add detailed metadata if requested
        if include_full_metadata:
            final_response['detailed_metadata'] = {
                'parsing_results': parsing_result.get('validation_results', {}),
                'prompt_metadata': parsing_result['metadata']['generation_metadata'].get('prompt_metadata', {}),
                'context_stats': formatted_context['summary'],
                'generation_details': parsing_result['metadata']['generation_metadata']
            }
        
        return {
            'final_response': final_response,
            'metadata': {
                'pipeline_summary': {
                    'total_steps_completed': 5,
                    'validation_passed': parsed_response['validation_passed'],
                    'quality_score': parsed_response['quality_score'],
                    'sources_used': len(citations),
                    'response_ready': True
                },
                'export_metadata': response_metadata
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'final_response': {},
            'metadata': {
                'pipeline_summary': {
                    'total_steps_completed': 4,  # Failed at final step
                    'validation_passed': False,
                    'quality_score': 0.0,
                    'sources_used': 0,
                    'response_ready': False
                }
            },
            'success': False,
            'error': f"Metadata extraction failed: {str(e)}"
        }


#@step
def format_final_cli_output(
    final_result: Dict[str, Any],
    output_format: str = "detailed"
) -> Dict[str, Any]:
    """
    Format the final response for CLI display.
    
    Args:
        final_result: Output from extract_metadata_and_citations step
        output_format: Format type ('detailed', 'simple', 'json')
        
    Returns:
        Dictionary with formatted output for CLI
    """
    try:
        if not final_result['success']:
            return {
                'cli_output': f"❌ Generation failed: {final_result['error']}",
                'display_metadata': {},
                'success': False,
                'error': final_result['error']
            }
        
        final_response = final_result['final_response']
        
        if output_format == "simple":
            cli_output = final_response['response']
            display_metadata = {
                'model': final_response['metadata']['generation_info']['model_used'],
                'sources': len(final_response['citations'])
            }
            
        elif output_format == "json":
            import json
            cli_output = json.dumps(final_response, indent=2)
            display_metadata = final_response['metadata']
            
        else:  # detailed format
            # Build detailed CLI output
            cli_parts = []
            cli_parts.append("📋 CLINICAL RESPONSE:")
            cli_parts.append("=" * 60)
            cli_parts.append(final_response['response'])
            cli_parts.append("=" * 60)
            
            # Generation details
            gen_info = final_response['metadata']['generation_info']
            cli_parts.append("\n📊 Generation Details:")
            cli_parts.append(f"   Model: {gen_info['model_used']}")
            cli_parts.append(f"   Quality Score: {final_response['metadata']['content_info']['quality_score']:.2f}")
            cli_parts.append(f"   Generation Time: {gen_info['generation_time']:.2f}s")
            if gen_info['total_tokens'] > 0:
                cli_parts.append(f"   Tokens: {gen_info['total_tokens']}")
            
            # Evidence sources
            cli_parts.append("\n📚 Evidence Sources:")
            for i, citation in enumerate(final_response['citations'], 1):
                cli_parts.append(f"   {i}. {citation['document']} ({citation['section']}) - Relevance: {citation['relevance']:.3f}")
            
            cli_output = "\n".join(cli_parts)
            display_metadata = final_response['metadata']
        
        return {
            'cli_output': cli_output,
            'display_metadata': display_metadata,
            'metadata': {
                'output_format': output_format,
                'response_length': len(cli_output),
                'final_metadata': final_result.get('metadata', {})
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'cli_output': f"❌ Output formatting failed: {str(e)}",
            'display_metadata': {},
            'metadata': {
                'output_format': output_format,
                'formatting_error': str(e)
            },
            'success': False,
            'error': f"CLI formatting failed: {str(e)}"
        }