"""Text preprocessing step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path
import re

from zenml import step


#@step
def preprocess_text(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and preprocess extracted text using basic cleaning.
    
    Args:
        extraction_result: Output from extract_text_from_pdf step
        
    Returns:
        Dictionary containing cleaned text and metadata
    """
    if not extraction_result['success']:
        return extraction_result
    
    try:
        text = extraction_result['text']
        metadata = extraction_result['metadata']
        
        return {
            'text': text,
            'metadata': {
                **metadata,
                'preprocessing_applied': True,
                'original_length': len(extraction_result['text']),
                'cleaned_length': len(text)
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'text': extraction_result['text'],
            'metadata': extraction_result['metadata'],
            'success': False,
            'error': f"Preprocessing failed: {str(e)}"
        }