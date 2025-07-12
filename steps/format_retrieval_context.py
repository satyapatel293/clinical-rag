"""Format retrieval context step for Clinical RAG generation pipeline."""

from typing import Dict, Any, List
from zenml import step


#@step
def format_retrieval_context(
    query: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format retrieved chunks and query for LLM consumption.
    
    Args:
        query: User's clinical question
        retrieved_chunks: List of retrieved chunks with metadata from search
        
    Returns:
        Dictionary containing formatted context and metadata
    """
    try:
        if not retrieved_chunks:
            return {
                'formatted_context': {},
                'success': False,
                'error': 'No retrieved chunks provided for context formatting'
            }
        
        # Format each chunk with source information
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            formatted_chunk = {
                'source_id': i,
                'content': chunk.get('text', ''),
                'similarity': chunk.get('similarity', 0.0),
                'section_type': chunk.get('section_type', 'unknown'),
                'document_name': chunk.get('filename', 'unknown'),
                'chunk_metadata': {
                    'similarity_score': chunk.get('similarity', 0.0),
                    'section': chunk.get('section_type', 'unknown'),
                    'document': chunk.get('filename', 'unknown')
                }
            }
            formatted_chunks.append(formatted_chunk)
        
        # Sort by similarity (highest first)
        formatted_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Create context summary
        context_summary = {
            'total_chunks': len(formatted_chunks),
            'similarity_range': {
                'highest': max(chunk['similarity'] for chunk in formatted_chunks),
                'lowest': min(chunk['similarity'] for chunk in formatted_chunks),
                'average': sum(chunk['similarity'] for chunk in formatted_chunks) / len(formatted_chunks)
            },
            'sections_covered': list(set(chunk['section_type'] for chunk in formatted_chunks)),
            'documents_referenced': list(set(chunk['document_name'] for chunk in formatted_chunks))
        }
        
        # Format context for LLM prompt
        context_parts = []
        for chunk in formatted_chunks:
            context_parts.append(
                f"[Source {chunk['source_id']} - {chunk['section_type'].title()} - "
                f"Relevance: {chunk['similarity']:.2f}]\n{chunk['content']}"
            )
        
        formatted_context = {
            'query': query,
            'context_text': "\n\n".join(context_parts),
            'chunks': formatted_chunks,
            'summary': context_summary
        }
        
        return {
            'formatted_context': formatted_context,
            'metadata': {
                'input_query': query,
                'chunks_processed': len(retrieved_chunks),
                'context_length': len(formatted_context['context_text']),
                'similarity_stats': context_summary['similarity_range'],
                'sections_included': context_summary['sections_covered'],
                'documents_included': context_summary['documents_referenced']
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'formatted_context': {},
            'metadata': {
                'input_query': query if 'query' in locals() else 'unknown',
                'chunks_processed': len(retrieved_chunks) if retrieved_chunks else 0
            },
            'success': False,
            'error': f"Context formatting failed: {str(e)}"
        }