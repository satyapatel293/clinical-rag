"""Text preprocessing step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path
import re

from zenml import step


@step
def preprocess_text(extraction_result: Dict[str, Any], save_temp: bool = True) -> Dict[str, Any]:
    """
    Clean and preprocess extracted text using basic cleaning.
    
    Args:
        extraction_result: Output from extract_text_from_pdf step
        save_temp: Whether to save processed text to temp directory
        
    Returns:
        Dictionary containing cleaned text and metadata
    """
    if not extraction_result['success']:
        return extraction_result
    
    try:
        text = extraction_result['text']
        metadata = extraction_result['metadata']
        
        # Very basic cleaning - keep almost everything
        # Remove page markers (our own format)
        text = re.sub(r'=== Page \d+ ===\s*', '', text)
        
        # Basic whitespace cleanup
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = text.strip()
        
        # Save to processed directory if requested
        if save_temp:
            from utils.config import PROCESSED_DATA_DIR
            processed_dir = Path(PROCESSED_DATA_DIR)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = processed_dir / f"{Path(metadata['file_path']).stem}_processed.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            metadata['processed_temp_file'] = str(output_file)
        
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