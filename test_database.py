"""Test database connection and operations."""

import os
from src.database import db_manager

def main():
    """Test database connectivity and basic operations."""
    print("Testing Clinical RAG Database Connection...")
    
    # Test connection
    result = db_manager.test_connection()
    
    if result['success']:
        print("✅ Database connection successful!")
        print(f"📊 PostgreSQL version: {result['postgresql_version']}")
        print(f"🔢 pgvector test distance: {result['pgvector_test_distance']}")
        
        # Get statistics
        stats = db_manager.get_statistics()
        print(f"\n📈 Database Statistics:")
        print(f"   Documents: {stats['documents']['total_docs']} total, {stats['documents']['completed_docs']} completed")
        print(f"   Chunks: {stats['chunks']['total_chunks']} total")
        print(f"   Embeddings: {len(stats['embeddings'])} model(s)")
        for embedding_stat in stats['embeddings']:
            print(f"     - {embedding_stat['model_name']}: {embedding_stat['total_embeddings']} embeddings")
            
    else:
        print("❌ Database connection failed!")
        print(f"   Error: {result['error']}")

if __name__ == "__main__":
    main()