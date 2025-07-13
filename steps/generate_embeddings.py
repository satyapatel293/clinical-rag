"""Embedding generation step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path
import numpy as np

from zenml import step


@step
def generate_embeddings(
    chunking_result: Dict[str, Any],
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Generate embeddings for text chunks using sentence transformers.
    
    Args:
        chunking_result: Output from chunk_text step
        model_name: Sentence transformer model to use
        
    Returns:
        Dictionary containing embeddings and metadata
    """
    if not chunking_result['success']:
        return {
            'embeddings': [],
            'metadata': chunking_result['metadata'],
            'success': False,
            'error': f"Cannot generate embeddings - chunking failed: {chunking_result['error']}"
        }
    
    try:
        # Import here to avoid loading model unnecessarily
        from sentence_transformers import SentenceTransformer
        
        chunks = chunking_result['chunks']
        metadata = chunking_result['metadata']
        
        print(f"Loading sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        # Generate embeddings in batches for efficiency
        embeddings_array = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Combine embeddings with chunk metadata
        chunk_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_array)):
            chunk_embeddings.append({
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text'],
                'section_type': chunk['section_type'],
                'embedding': embedding,
                'embedding_dim': len(embedding),
                'source_file': chunk['source_file']
            })
        
        return {
            'embeddings': chunk_embeddings,
            'metadata': {
                **metadata,
                'embedding_model': model_name,
                'embedding_dim': len(embeddings_array[0]),
                'total_embeddings': len(chunk_embeddings)
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'embeddings': [],
            'metadata': chunking_result['metadata'],
            'success': False,
            'error': f"Embedding generation failed: {str(e)}"
        }