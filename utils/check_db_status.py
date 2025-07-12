#!/usr/bin/env python3
"""
Database status checker for Clinical RAG system.
Run this script to see the current contents and statistics of your database.
"""

from utils.database import DatabaseManager
import json

def main():
    db = DatabaseManager()
    
    print('=== DATABASE STATISTICS ===')
    stats = db.get_statistics()
    print(f'Documents: {stats["documents"]}')
    print(f'Chunks: {stats["chunks"]}')
    print(f'Embeddings: {stats["embeddings"]}')

    print('\n=== DOCUMENTS TABLE ===')
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id, filename, title, num_pages, processing_status, created_at FROM documents ORDER BY id;')
            docs = cur.fetchall()
            for doc in docs:
                print(f'ID: {doc[0]}, File: {doc[1]}, Title: {doc[2]}, Pages: {doc[3]}, Status: {doc[4]}, Created: {doc[5]}')

    print('\n=== CHUNKS SAMPLE (first 5) ===')
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id, document_id, chunk_id, section_type, char_length, substring(text, 1, 100) as text_preview FROM chunks ORDER BY id LIMIT 5;')
            chunks = cur.fetchall()
            for chunk in chunks:
                print(f'ID: {chunk[0]}, Doc: {chunk[1]}, Chunk: {chunk[2]}, Section: {chunk[3]}, Length: {chunk[4]}')
                print(f'Preview: {chunk[5]}...')
                print()

    print('=== SECTION TYPE DISTRIBUTION ===')
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT section_type, COUNT(*) FROM chunks GROUP BY section_type ORDER BY COUNT(*) DESC;')
            sections = cur.fetchall()
            for section in sections:
                print(f'{section[0]}: {section[1]} chunks')

    print('\n=== EMBEDDING DIMENSIONS (check one embedding) ===')
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT embedding FROM embeddings LIMIT 1;')
            result = cur.fetchone()
            if result:
                embedding = result[0]
                print(f'Embedding dimensions: {len(embedding)}')
                print(f'First 5 values: {embedding[:5]}')

    print('\n=== CHUNK LENGTH DISTRIBUTION ===')
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT MIN(char_length), MAX(char_length), AVG(char_length) FROM chunks;')
            result = cur.fetchone()
            if result:
                print(f'Min length: {result[0]}, Max length: {result[1]}, Avg length: {result[2]:.1f}')

    print('\n=== EMBEDDINGS SAMPLE (first 3) ===')
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id, chunk_id, model_name FROM embeddings ORDER BY id LIMIT 3;')
            embeddings = cur.fetchall()
            for emb in embeddings:
                print(f'ID: {emb[0]}, Chunk: {emb[1]}, Model: {emb[2]}')

if __name__ == "__main__":
    main()