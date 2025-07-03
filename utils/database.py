"""Database connection and operations for Clinical RAG system."""

import os
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import json

from .config import DATABASE_URL


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return status."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    
                    # Test pgvector
                    cur.execute("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector;")
                    distance = cur.fetchone()[0]
                    
                    return {
                        'success': True,
                        'postgresql_version': version,
                        'pgvector_test_distance': distance,
                        'error': None
                    }
        except Exception as e:
            return {
                'success': False,
                'postgresql_version': None,
                'pgvector_test_distance': None,
                'error': str(e)
            }
    
    def insert_document(self, document_data: Dict[str, Any]) -> int:
        """Insert a new document and return its ID."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                insert_sql = """
                    INSERT INTO documents (filename, file_path, title, author, subject, 
                                         num_pages, file_size_bytes, original_text_length, 
                                         processed_text_length, total_chunks, processing_status)
                    VALUES (%(filename)s, %(file_path)s, %(title)s, %(author)s, %(subject)s,
                            %(num_pages)s, %(file_size_bytes)s, %(original_text_length)s,
                            %(processed_text_length)s, %(total_chunks)s, %(processing_status)s)
                    RETURNING id;
                """
                cur.execute(insert_sql, document_data)
                document_id = cur.fetchone()[0]
                conn.commit()
                return document_id
    
    def insert_chunks(self, document_id: int, chunks: List[Dict[str, Any]]) -> int:
        """Insert chunks for a document and return count inserted."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                insert_sql = """
                    INSERT INTO chunks (document_id, chunk_id, text, section_type, 
                                      sentence_start, sentence_end, char_length)
                    VALUES (%(document_id)s, %(chunk_id)s, %(text)s, %(section_type)s,
                            %(sentence_start)s, %(sentence_end)s, %(char_length)s)
                    RETURNING id;
                """
                
                chunk_ids = []
                for chunk in chunks:
                    chunk_data = {
                        'document_id': document_id,
                        'chunk_id': chunk['chunk_id'],
                        'text': chunk['text'],
                        'section_type': chunk['section_type'],
                        'sentence_start': chunk['sentence_start'],
                        'sentence_end': chunk['sentence_end'],
                        'char_length': len(chunk['text'])
                    }
                    cur.execute(insert_sql, chunk_data)
                    chunk_ids.append(cur.fetchone()[0])
                
                conn.commit()
                return len(chunk_ids)
    
    def insert_embeddings(self, chunk_embeddings: List[Dict[str, Any]]) -> int:
        """Insert embeddings for chunks and return count inserted."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                insert_sql = """
                    INSERT INTO embeddings (chunk_id, embedding, model_name, model_version)
                    VALUES (%(chunk_id)s, %(embedding)s, %(model_name)s, %(model_version)s);
                """
                
                for embedding_data in chunk_embeddings:
                    # Convert numpy array to list for PostgreSQL
                    embedding_list = embedding_data['embedding'].tolist()
                    data = {
                        'chunk_id': embedding_data['chunk_id'],
                        'embedding': embedding_list,
                        'model_name': embedding_data['model_name'],
                        'model_version': embedding_data.get('model_version', '1.0')
                    }
                    cur.execute(insert_sql, data)
                
                conn.commit()
                return len(chunk_embeddings)
    
    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get document by filename."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE filename = %s;", (filename,))
                result = cur.fetchone()
                return dict(result) if result else None
    
    def search_similar_chunks(self, query_embedding: List[float], 
                            similarity_threshold: float = 0.7, 
                            max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM search_similar_chunks(%s, %s, %s);
                """, (query_embedding, similarity_threshold, max_results))
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def update_document_status(self, document_id: int, status: str, error_message: str = None):
        """Update document processing status."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents 
                    SET processing_status = %s, error_message = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s;
                """, (status, error_message, document_id))
                conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Documents stats
                cur.execute("SELECT COUNT(*) as total_docs, COUNT(*) FILTER (WHERE processing_status = 'completed') as completed_docs FROM documents;")
                doc_stats = cur.fetchone()
                
                # Chunks stats
                cur.execute("SELECT COUNT(*) as total_chunks FROM chunks;")
                chunk_stats = cur.fetchone()
                
                # Embeddings stats
                cur.execute("SELECT COUNT(*) as total_embeddings, model_name FROM embeddings GROUP BY model_name;")
                embedding_stats = cur.fetchall()
                
                return {
                    'documents': dict(doc_stats),
                    'chunks': dict(chunk_stats),
                    'embeddings': [dict(row) for row in embedding_stats]
                }


# Global database manager instance
db_manager = DatabaseManager()