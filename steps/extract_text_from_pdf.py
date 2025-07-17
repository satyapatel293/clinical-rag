"""PDF text extraction step for Clinical RAG system."""

from typing import Dict, Any
from pathlib import Path

from zenml import step
import fitz  # PyMuPDF


def detect_columns_with_text_blocks(page, footer_margin=50):
    """
    Detect column boundaries using PyMuPDF's text block detection.
    
    Args:
        page: PyMuPDF page object
        footer_margin: Height of footer margin to ignore
        
    Returns:
        List of column bounding boxes
    """
    try:
        # Get text blocks
        blocks = page.get_text("blocks")
        
        # Filter out footer text and empty blocks
        page_height = page.rect.height
        filtered_blocks = []
        
        for block in blocks:
            if len(block) >= 5:  # Ensure block has text content
                x0, y0, x1, y1, text = block[:5]
                # Skip if in footer area or empty
                if y1 < (page_height - footer_margin) and text.strip():
                    filtered_blocks.append({
                        'bbox': fitz.Rect(x0, y0, x1, y1),
                        'text': text.strip()
                    })
        
        if not filtered_blocks:
            return []
        
        # Sort blocks by position (top to bottom, left to right)
        filtered_blocks.sort(key=lambda b: (b['bbox'].y0, b['bbox'].x0))
        
        # Group blocks into columns based on x-coordinates
        columns = []
        page_width = page.rect.width
        
        # Simple column detection: group blocks by their x-center position
        left_blocks = []
        right_blocks = []
        
        for block in filtered_blocks:
            x_center = (block['bbox'].x0 + block['bbox'].x1) / 2
            if x_center < page_width / 2:
                left_blocks.append(block)
            else:
                right_blocks.append(block)
        
        # Create column bounding boxes
        if left_blocks:
            left_x0 = min(b['bbox'].x0 for b in left_blocks)
            left_y0 = min(b['bbox'].y0 for b in left_blocks)
            left_x1 = max(b['bbox'].x1 for b in left_blocks)
            left_y1 = max(b['bbox'].y1 for b in left_blocks)
            columns.append(fitz.Rect(left_x0, left_y0, left_x1, left_y1))
        
        if right_blocks:
            right_x0 = min(b['bbox'].x0 for b in right_blocks)
            right_y0 = min(b['bbox'].y0 for b in right_blocks)
            right_x1 = max(b['bbox'].x1 for b in right_blocks)
            right_y1 = max(b['bbox'].y1 for b in right_blocks)
            columns.append(fitz.Rect(right_x0, right_y0, right_x1, right_y1))
        
        return columns
        
    except Exception as e:
        print(f"Error in column detection: {e}")
        return []


@step
def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from PDF using PyMuPDF with automatic column detection.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extract metadata
        pdf_metadata = doc.metadata
        metadata = {
            'file_path': pdf_path,
            'filename': Path(pdf_path).name,
            'num_pages': len(doc),
            'title': pdf_metadata.get('title', '') if pdf_metadata else '',
            'author': pdf_metadata.get('author', '') if pdf_metadata else '',
            'subject': pdf_metadata.get('subject', '') if pdf_metadata else ''
        }
        
        # Extract all pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Detect columns using text block analysis
            column_boxes = detect_columns_with_text_blocks(page)
            
            if column_boxes:
                # Extract text from detected columns
                page_text = ""
                for i, column_rect in enumerate(column_boxes):
                    column_text = page.get_text(clip=column_rect)
                    if column_text.strip():
                        column_label = f"COLUMN {i + 1}" if len(column_boxes) > 1 else "TEXT"
                        page_text += f"{column_label}:\n{column_text.strip()}\n\n"
                
                if page_text.strip():
                    text += f"--- Page {page_num + 1} ---\n{page_text}\n"
            else:
                # Fallback to full page extraction if no columns detected
                full_text = page.get_text()
                if full_text.strip():
                    text += f"--- Page {page_num + 1} ---\n{full_text.strip()}\n\n"
        
        # Close the document
        doc.close()
        
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