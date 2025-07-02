"""Main pipeline for Clinical RAG system."""

from zenml import pipeline
from .steps import extract_text_from_pdf, preprocess_text, chunk_text


@pipeline
def pdf_processing_pipeline(pdf_path: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Pipeline to extract, preprocess, and chunk text from PDF.
    
    Args:
        pdf_path: Path to PDF file to process
        chunk_size: Target size for each chunk (characters)
        overlap: Number of characters to overlap between chunks
    """
    extraction_result = extract_text_from_pdf(pdf_path)
    processed_result = preprocess_text(extraction_result)
    chunking_result = chunk_text(processed_result, chunk_size, overlap)
    return chunking_result