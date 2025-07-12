"""Embedding storage step for Clinical RAG system."""

from typing import Dict, Any
import os

from zenml import step
from utils.database import DatabaseManager


#@step
def store_embeddings_in_database(
    embedding_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Store embeddings and chunks in PostgreSQL database.
    
    Args:
        embedding_result: Output from generate_embeddings step
        
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
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/clinical_rag')
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