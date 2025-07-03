"""Build clinical prompt step for Clinical RAG generation pipeline."""

from typing import Dict, Any
from zenml import step


@step
def build_clinical_prompt(
    formatting_result: Dict[str, Any],
    prompt_template: str = "clinical_rag_default"
) -> Dict[str, Any]:
    """
    Build clinical prompt with retrieved context for LLM generation.
    
    Args:
        formatting_result: Output from format_retrieval_context step
        prompt_template: Template to use for prompt construction
        
    Returns:
        Dictionary containing complete prompt and metadata
    """
    try:
        if not formatting_result['success']:
            return {
                'prompt': "",
                'success': False,
                'error': f"Cannot build prompt - context formatting failed: {formatting_result['error']}"
            }
        
        formatted_context = formatting_result['formatted_context']
        query = formatted_context['query']
        context_text = formatted_context['context_text']
        context_summary = formatted_context['summary']
        
        # Build the clinical prompt
        prompt = f"""You are a clinical decision support assistant for orthopedic physical therapy. Your role is to provide evidence-based guidance to healthcare professionals.

CLINICAL CONTEXT:
{context_text}

CLINICAL QUERY: {query}

INSTRUCTIONS:
1. Provide a clear, evidence-based response based ONLY on the provided clinical context
2. If the context doesn't fully address the query, clearly state what information is missing
3. Include relevant clinical considerations and contraindications when applicable
4. Use professional medical terminology appropriate for healthcare providers
5. Structure your response with clear sections (Assessment, Recommendations, Considerations)
6. If multiple sources conflict, acknowledge the discrepancy
7. Do not provide information not supported by the given context

IMPORTANT: This is decision support for healthcare professionals, not direct patient care advice.

RESPONSE:"""
        
        # Calculate prompt statistics
        prompt_stats = {
            'total_length': len(prompt),
            'context_length': len(context_text),
            'query_length': len(query),
            'instruction_length': len(prompt) - len(context_text) - len(query),
            'estimated_tokens': len(prompt.split()),  # Rough token estimate
            'sources_included': context_summary['total_chunks'],
            'sections_covered': len(context_summary['sections_covered']),
            'documents_referenced': len(context_summary['documents_referenced'])
        }
        
        return {
            'prompt': prompt,
            'metadata': {
                'prompt_template': prompt_template,
                'query': query,
                'context_stats': context_summary,
                'prompt_stats': prompt_stats,
                'sources_metadata': [
                    {
                        'source_id': chunk['source_id'],
                        'similarity': chunk['similarity'],
                        'section': chunk['section_type'],
                        'document': chunk['document_name']
                    }
                    for chunk in formatted_context['chunks']
                ]
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'prompt': "",
            'metadata': {
                'prompt_template': prompt_template,
                'context_available': formatting_result.get('success', False)
            },
            'success': False,
            'error': f"Prompt building failed: {str(e)}"
        }


@step 
def build_simple_clinical_prompt(
    query: str,
    prompt_template: str = "clinical_simple"
) -> Dict[str, Any]:
    """
    Build simple clinical prompt without context for testing/fallback.
    
    Args:
        query: Clinical question
        prompt_template: Template to use for prompt construction
        
    Returns:
        Dictionary containing simple prompt and metadata
    """
    try:
        prompt = f"""You are a clinical decision support assistant for orthopedic physical therapy. 

QUERY: {query}

Provide a brief, evidence-based response. Note that this response is generated without specific clinical context and should be verified with current clinical guidelines.

RESPONSE:"""
        
        prompt_stats = {
            'total_length': len(prompt),
            'query_length': len(query),
            'estimated_tokens': len(prompt.split()),
            'context_included': False
        }
        
        return {
            'prompt': prompt,
            'metadata': {
                'prompt_template': prompt_template,
                'query': query,
                'prompt_stats': prompt_stats,
                'context_used': False
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'prompt': "",
            'metadata': {
                'prompt_template': prompt_template,
                'query': query if 'query' in locals() else 'unknown'
            },
            'success': False,
            'error': f"Simple prompt building failed: {str(e)}"
        }