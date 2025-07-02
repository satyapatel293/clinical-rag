"""ZenML pipeline steps for Clinical RAG system."""

from typing import Dict, Any, List
from pathlib import Path
import re
import json

from zenml import step
import pdfplumber
from unstructured.cleaners.core import clean_extra_whitespace, clean_non_ascii_chars


@step
def extract_text_from_pdf(pdf_path: str, save_temp: bool = True) -> Dict[str, Any]:
    """
    Extract text content from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        save_temp: Whether to save extracted text to temp directory
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract metadata
            metadata = {
                'file_path': pdf_path,
                'filename': Path(pdf_path).name,
                'num_pages': len(pdf.pages),
                'title': pdf.metadata.get('Title', '') if pdf.metadata else '',
                'author': pdf.metadata.get('Author', '') if pdf.metadata else '',
                'subject': pdf.metadata.get('Subject', '') if pdf.metadata else ''
            }
            
            # Extract text from all pages with better formatting
            pages_text = []
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(f"=== Page {page_num + 1} ===\n{page_text.strip()}")
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            full_text = "\n\n".join(pages_text)
            
            # Save to temp directory if requested
            if save_temp:
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                output_file = temp_dir / f"{Path(pdf_path).stem}_extracted.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                
                metadata['temp_file'] = str(output_file)
            
            return {
                'text': full_text,
                'metadata': metadata,
                'success': True,
                'error': None
            }
            
    except Exception as e:
        return {
            'text': '',
            'metadata': {'file_path': pdf_path, 'filename': Path(pdf_path).name},
            'success': False,
            'error': str(e)
        }


@step
def preprocess_text(extraction_result: Dict[str, Any], save_temp: bool = True) -> Dict[str, Any]:
    """
    Clean and preprocess extracted text using unstructured library.
    
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
        
        # Remove page markers (our own format)
        text = re.sub(r'=== Page \d+ ===\s*', '', text)
        
        # Use unstructured cleaners
        text = clean_extra_whitespace(text)
        text = clean_non_ascii_chars(text)
        
        # Additional clinical document cleaning
        # Remove common header/footer patterns
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, page numbers, copyright notices
            if (not line or 
                re.match(r'^\d+$', line) or 
                'copyright' in line.lower() or
                '©' in line or
                len(line) < 3):
                continue
            cleaned_lines.append(line)
        
        # Rejoin and final cleanup
        text = '\n'.join(cleaned_lines)
        
        # Fix sentence spacing
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        # Save to temp directory if requested
        if save_temp:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            output_file = temp_dir / f"{Path(metadata['file_path']).stem}_processed.txt"
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


@step
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
        
        # Simple sentence-based chunking with overlap
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            # Add sentence if it fits
            if len(current_chunk + sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': len(chunks),
                        'sentence_start': chunk_start,
                        'sentence_end': i - 1,
                        'section_type': _identify_section_type(current_chunk),
                        'source_file': metadata['filename']
                    })
                
                # Start new chunk with overlap
                overlap_sentences = sentences[max(0, i - 2):i]  # Previous 2 sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                chunk_start = max(0, i - 2)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': len(chunks),
                'sentence_start': chunk_start,
                'sentence_end': len(sentences) - 1,
                'section_type': _identify_section_type(current_chunk),
                'source_file': metadata['filename']
            })
        
        # Save chunks to temp directory if requested
        if save_temp:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            chunks_file = temp_dir / f"{Path(metadata['file_path']).stem}_chunks.json"
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