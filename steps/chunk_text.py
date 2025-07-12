"""Text chunking step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path
import re
import json

from zenml import step


def _is_reference_sentence(sentence: str) -> bool:
    """Check if a sentence is primarily a reference or citation."""
    sentence = sentence.strip()
    
    # Only filter obvious references
    obvious_references = [
        r'doi:',    # Contains DOI
        r'https?://',  # Contains URL
        r'^\d+\.\s*[A-Z][a-z]+.*\d{4}',  # Numbered reference with year
    ]
    
    for pattern in obvious_references:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    
    # Check if sentence is mostly just a URL or DOI
    if len(sentence) < 50 and ('doi' in sentence.lower() or 'http' in sentence.lower()):
        return True
    
    return False


def _is_meaningful_sentence(sentence: str) -> bool:
    """Check if a sentence contains meaningful content."""
    sentence = sentence.strip()
    
    if len(sentence) < 15:  # Too short
        return False
    
    # Skip if it's mostly references
    if _is_reference_sentence(sentence):
        return False
    
    # Check if sentence has enough alphabetic content
    words = sentence.split()
    alphabetic_words = [w for w in words if w.isalpha() and len(w) > 2]
    
    # Sentence should have enough real words
    if len(words) < 3 or len(alphabetic_words) < 2:
        return False
        
    # Very basic meaningfulness check - not too strict
    return True


def _identify_section_type(text: str) -> str:
    """Identify the type of content in a chunk."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['introduction', 'background', 'overview']):
        return 'introduction'
    elif any(word in text_lower for word in ['method', 'procedure', 'technique']):
        return 'methods'
    elif any(word in text_lower for word in ['diagnosis', 'assessment', 'evaluation']):
        return 'diagnosis'
    elif any(word in text_lower for word in ['treatment', 'intervention', 'therapy']):
        return 'treatment'
    elif any(word in text_lower for word in ['conclusion', 'summary', 'recommendation']):
        return 'conclusion'
    elif any(word in text_lower for word in ['reference', 'bibliography', 'citation']):
        return 'references'
    else:
        return 'general'


#@step
def chunk_text(
    processed_result: Dict[str, Any], 
    chunk_size: int = 1000, 
    overlap: int = 200,
    save_temp: bool = True
) -> Dict[str, Any]:
    """
    Split processed text into chunks with section boundary respect and overlap.
    
    Args:
        processed_result: Output from preprocess_text step
        chunk_size: Target size for each chunk (characters)
        overlap: Number of characters to overlap between chunks
        save_temp: Whether to save chunks to temp directory
        
    Returns:
        Dictionary containing chunks and metadata
    """
    if not processed_result['success']:
        return {
            'chunks': [],
            'metadata': processed_result['metadata'],
            'success': False,
            'error': f"Cannot chunk - preprocessing failed: {processed_result['error']}"
        }
    
    try:
        text = processed_result['text']
        metadata = processed_result['metadata']
        
        # Improved sentence-based chunking with quality filtering
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            # Skip sentences that are mostly references or citations
            if _is_reference_sentence(sentence):
                continue
                
            # Only add meaningful sentences
            if not _is_meaningful_sentence(sentence):
                continue
            
            # Add sentence if it fits
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it has substantial content
                if len(current_chunk.strip()) > 150:  # Minimum chunk size
                    chunk_text = current_chunk.strip()
                    chunks.append({
                        'text': chunk_text,
                        'chunk_id': len(chunks),
                        'sentence_start': chunk_start,
                        'sentence_end': i - 1,
                        'section_type': _identify_section_type(chunk_text),
                        'source_file': metadata['filename']
                    })
                
                # Start new chunk with overlap
                overlap_sentences = sentences[max(0, i - 2):i]  # Previous 2 sentences  
                meaningful_overlap = [s for s in overlap_sentences if _is_meaningful_sentence(s) and not _is_reference_sentence(s)]
                current_chunk = " ".join(meaningful_overlap) + " " + sentence + " "
                chunk_start = max(0, i - 2)
        
        # Add final chunk
        if len(current_chunk.strip()) > 150:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': len(chunks),
                'sentence_start': chunk_start,
                'sentence_end': len(sentences) - 1,
                'section_type': _identify_section_type(current_chunk),
                'source_file': metadata['filename']
            })
        
        # Save chunks to processed directory if requested
        if save_temp:
            from utils.config import PROCESSED_DATA_DIR
            processed_dir = Path(PROCESSED_DATA_DIR)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            chunks_file = processed_dir / f"{Path(metadata['file_path']).stem}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            metadata['chunks_temp_file'] = str(chunks_file)
        
        return {
            'chunks': chunks,
            'metadata': {
                **metadata,
                'total_chunks': len(chunks),
                'chunk_size': chunk_size,
                'overlap': overlap
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'chunks': [],
            'metadata': processed_result['metadata'],
            'success': False,
            'error': str(e)
        }