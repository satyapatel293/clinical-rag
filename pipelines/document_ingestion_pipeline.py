"""Main Clinical RAG pipeline for processing PDFs into searchable embeddings."""

from zenml import pipeline
from steps.extract_text_from_pdf import extract_text_from_pdf
from steps.preprocess_text import preprocess_text
from steps.chunk_text import chunk_text
from steps.generate_embeddings import generate_embeddings
from steps.store_embeddings import store_embeddings_in_database


@pipeline(enable_cache=False)
def pdf_processing_pipeline(
    pdf_path: str, 
    chunk_size: int = 1000, 
    overlap: int = 200, 
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    Pipeline to extract, preprocess, chunk, embed, and store text from PDF.
    
    Args:
        pdf_path: Path to PDF file to process
        chunk_size: Target size for each chunk (characters)
        overlap: Number of characters to overlap between chunks
        model_name: Sentence transformer model for embeddings
    """
    extraction_result = extract_text_from_pdf(pdf_path)
    processed_result = preprocess_text(extraction_result)
    chunking_result = chunk_text(processed_result, chunk_size, overlap)
    embedding_result = generate_embeddings(chunking_result, model_name)
    storage_result = store_embeddings_in_database(embedding_result)
    return storage_result