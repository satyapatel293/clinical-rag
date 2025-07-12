"""PDF text extraction step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path

from zenml import step
import pdfplumber


#@step
def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text content from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
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
            
            # Extract text from all pages
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return {
                'text': text.strip(),
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