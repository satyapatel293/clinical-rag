"""ZenML pipeline steps for Clinical RAG system."""

from typing import Dict, Any, List
from pathlib import Path
import re
import json
import os

from zenml import step
import pdfplumber
from unstructured.cleaners.core import clean_extra_whitespace, clean_non_ascii_chars
import numpy as np
from .database import DatabaseManager


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


@step
def generate_embeddings(
    chunking_result: Dict[str, Any],
    model_name: str = "all-MiniLM-L6-v2",
    save_temp: bool = True
) -> Dict[str, Any]:
    """
    Generate embeddings for text chunks using sentence transformers.
    
    Args:
        chunking_result: Output from chunk_text step
        model_name: Sentence transformer model to use
        save_temp: Whether to save embeddings to temp directory
        
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
        
        # Save embeddings to temp directory if requested
        if save_temp:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            embeddings_file = temp_dir / f"{Path(metadata['file_path']).stem}_embeddings.npz"
            
            # Save as compressed numpy arrays for efficiency
            embeddings_dict = {
                f"embedding_{i}": emb['embedding'] 
                for i, emb in enumerate(chunk_embeddings)
            }
            np.savez_compressed(embeddings_file, **embeddings_dict)
            
            metadata['embeddings_temp_file'] = str(embeddings_file)
        
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


@step
def store_embeddings_in_database(
    embedding_result: Dict[str, Any],
    database_url: str = "postgresql://postgres:password@localhost:5432/clinical_rag"
) -> Dict[str, Any]:
    """
    Store embeddings and chunks in PostgreSQL database.
    
    Args:
        embedding_result: Output from generate_embeddings step
        database_url: Database connection URL (if None, uses environment variable)
        
    Returns:
        Dictionary containing storage results and metadata
    """
    if not embedding_result['success']:
        return {
            'stored_count': 0,
            'metadata': embedding_result['metadata'],
            'success': False,
            'error': f"Cannot store embeddings - generation failed: {embedding_result['error']}"
        }
    
    try:
        # Initialize database manager
        db_url = database_url or os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/clinical_rag')
        db_manager = DatabaseManager(db_url)
        
        embeddings = embedding_result['embeddings']
        metadata = embedding_result['metadata']
        
        print(f"Connecting to database and storing {len(embeddings)} embeddings...")
        
        # First, create or get document record
        document_data = {
            'filename': metadata['filename'],
            'file_path': metadata['file_path'],
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'num_pages': metadata.get('num_pages', 0),
            'file_size_bytes': 0,  # We could add this if needed
            'original_text_length': metadata.get('original_length', 0),
            'processed_text_length': metadata.get('cleaned_length', 0),
            'total_chunks': metadata.get('total_chunks', 0),
            'processing_status': 'completed'
        }
        
        document_id = db_manager.insert_document(document_data)
        print(f"Created/retrieved document record with ID: {document_id}")
        
        # Insert chunks and embeddings one by one to track IDs properly
        stored_count = 0
        chunk_embedding_pairs = []
        
        print(f"Inserting {len(embeddings)} chunks and embeddings...")
        
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                for embedding_data in embeddings:
                    try:
                        # Insert chunk first
                        chunk_insert_sql = """
                            INSERT INTO chunks (document_id, chunk_id, text, section_type, 
                                              sentence_start, sentence_end, char_length)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING id;
                        """
                        chunk_values = (
                            document_id,
                            embedding_data['chunk_id'], 
                            embedding_data['text'],
                            embedding_data['section_type'],
                            0,  # sentence_start
                            0,  # sentence_end  
                            len(embedding_data['text'])
                        )
                        cur.execute(chunk_insert_sql, chunk_values)
                        db_chunk_id = cur.fetchone()[0]
                        
                        # Insert embedding with the returned chunk ID
                        embedding_insert_sql = """
                            INSERT INTO embeddings (chunk_id, embedding, model_name, model_version)
                            VALUES (%s, %s, %s, %s);
                        """
                        embedding_values = (
                            db_chunk_id,
                            embedding_data['embedding'].tolist(),
                            metadata['embedding_model'],
                            '1.0'
                        )
                        cur.execute(embedding_insert_sql, embedding_values)
                        
                        stored_count += 1
                        
                        if stored_count % 50 == 0:
                            print(f"Stored {stored_count}/{len(embeddings)} embeddings...")
                            
                    except Exception as e:
                        print(f"Error storing chunk {embedding_data['chunk_id']}: {e}")
                        continue
                
                conn.commit()
        
        print(f"Successfully stored {stored_count} chunks and embeddings")
        
        # Get final statistics
        stats = db_manager.get_statistics()
        
        print(f"Successfully stored {stored_count} embeddings to database")
        print(f"Database now contains {stats['documents']['total_docs']} documents, {stats['chunks']['total_chunks']} chunks")
        total_embeddings = sum(e['total_embeddings'] for e in stats['embeddings']) if stats['embeddings'] else 0
        print(f"Total embeddings: {total_embeddings}")
        
        return {
            'stored_count': stored_count,
            'document_id': document_id,
            'database_stats': stats,
            'metadata': {
                **metadata,
                'database_url': db_url,
                'storage_success': True
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'stored_count': 0,
            'metadata': embedding_result['metadata'],
            'success': False,
            'error': f"Database storage failed: {str(e)}"
        }