"""PDF text extraction step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path

from zenml import step
import pdfplumber


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
            
            # Save to processed directory if requested
            if save_temp:
                from utils.config import PROCESSED_DATA_DIR
                processed_dir = Path(PROCESSED_DATA_DIR)
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = processed_dir / f"{Path(pdf_path).stem}_extracted.txt"
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