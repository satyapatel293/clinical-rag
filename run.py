#!/usr/bin/env python3
"""
Main CLI entry point for Clinical RAG system.
"""

import os
import sys
import argparse
from pathlib import Path

# Set environment variable for ZenML on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from pipelines.clinical_rag_pipeline import pdf_processing_pipeline
from utils.search import ClinicalRAGSearcher
from utils.generation import ClinicalResponseGenerator
from utils.database import DatabaseManager
from utils.config import DATA_DIR


def cmd_ingest(args):
    """Run the PDF ingestion pipeline."""
    pdf_path = args.pdf_path
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")
    print(f"Model: {args.model}")
    
    try:
        pipeline_run = pdf_processing_pipeline(
            pdf_path=str(pdf_path),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_name=args.model
        )
        
        # Get the result from the storage step
        step_output = pipeline_run.steps["store_embeddings_in_database"].output.load()
        
        if step_output['success']:
            print(f"✅ Successfully processed {Path(pdf_path).name}")
            print(f"📦 Stored {step_output['stored_count']} chunks and embeddings")
            stats = step_output['database_stats']
            total_embeddings = sum(e['total_embeddings'] for e in stats['embeddings']) if stats['embeddings'] else 0
            print(f"📊 Database now contains {stats['documents']['total_docs']} documents, {total_embeddings} total embeddings")
        else:
            print(f"❌ Processing failed: {step_output['error']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        sys.exit(1)


def cmd_search(args):
    """Run a search query."""
    try:
        searcher = ClinicalRAGSearcher()
        
        print(f"🔍 Searching for: {args.query}")
        print(f"Top {args.top_k} results (similarity ≥ {args.threshold})")
        
        results = searcher.search_similar_chunks(
            query=args.query,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            section_filter=args.section,
            enhance_query=not args.no_enhance
        )
        
        if results:
            print(f"\n=== Found {len(results)} results ===\n")
            for i, result in enumerate(results, 1):
                print(f"Result {i} (Similarity: {result['similarity']:.3f})")
                print(f"Document: {result['filename']}")
                print(f"Section: {result['section_type']}")
                
                text = result['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                
                print(f"Content: {text}")
                print("-" * 80)
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        sys.exit(1)


def cmd_ask(args):
    """Ask a clinical question using RAG (Retrieval + Generation)."""
    try:
        searcher = ClinicalRAGSearcher()
        
        print(f"🔍 Searching for: {args.query}")
        print(f"Top {args.top_k} results (similarity ≥ {args.threshold})")
        
        # Retrieve relevant chunks
        results = searcher.search_similar_chunks(
            query=args.query,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            section_filter=args.section,
            enhance_query=not args.no_enhance
        )
        
        if not results:
            print("No relevant clinical information found.")
            return
            
        print(f"✅ Found {len(results)} relevant chunks")
        print("🤖 Generating clinical response...\n")
        
        try:
            generator = ClinicalResponseGenerator()
            
            # Convert search results to format expected by generator
            chunks_for_generation = []
            for result in results:
                chunks_for_generation.append({
                    'content': result['text'],
                    'similarity': result['similarity'],
                    'section_type': result['section_type'],
                    'document_name': result['filename']
                })
            
            # Generate RAG response
            generation_result = generator.generate_response(args.query, chunks_for_generation)
            
            if generation_result['success']:
                print("📋 CLINICAL RESPONSE:")
                print("=" * 60)
                print(generation_result['response'])
                print("=" * 60)
                
                # Show metadata
                metadata = generation_result['metadata']
                print(f"\n📊 Generation Details:")
                print(f"   Model: {metadata['model']}")
                print(f"   Chunks used: {metadata['chunks_used']}")
                print(f"   Temperature: {metadata['temperature']}")
                if 'total_tokens' in metadata:
                    print(f"   Tokens: {metadata['total_tokens']}")
                
                # Show sources
                print(f"\n📚 Evidence Sources:")
                for i, source in enumerate(metadata['chunk_sources'], 1):
                    print(f"   {i}. {source['document']} ({source['section']}) - Relevance: {source['similarity']:.3f}")
                    
            else:
                print(f"❌ Generation failed: {generation_result['error']}")
                print("\n📄 Retrieved information (fallback):")
                _show_chunks(results)
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            print("\n📄 Retrieved information (fallback):")
            _show_chunks(results)
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        sys.exit(1)


def _show_chunks(results):
    """Helper function to display retrieved chunks."""
    for i, result in enumerate(results, 1):
        print(f"Result {i} (Similarity: {result['similarity']:.3f})")
        print(f"Document: {result['filename']}")
        print(f"Section: {result['section_type']}")
        
        text = result['text']
        if len(text) > 300:
            text = text[:300] + "..."
        
        print(f"Content: {text}")
        print("-" * 80)


def cmd_status(args):
    """Check database status."""
    try:
        db_manager = DatabaseManager()
        
        # Test connection
        status = db_manager.test_connection()
        if status['success']:
            print("✅ Database connection successful")
            print(f"PostgreSQL version: {status['postgresql_version']}")
            print(f"pgvector test distance: {status['pgvector_test_distance']}")
        else:
            print(f"❌ Database connection failed: {status['error']}")
            sys.exit(1)
        
        # Get statistics
        stats = db_manager.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"Documents: {stats['documents']['total_docs']} total, {stats['documents']['completed_docs']} completed")
        print(f"Chunks: {stats['chunks']['total_chunks']}")
        
        if stats['embeddings']:
            for embedding_stat in stats['embeddings']:
                print(f"Embeddings ({embedding_stat['model_name']}): {embedding_stat['total_embeddings']}")
        else:
            print("Embeddings: 0")
            
    except Exception as e:
        print(f"❌ Status check error: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Clinical RAG system - Process clinical PDFs and search semantically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF
  python run.py ingest data/raw/Achilles_Pain.pdf
  
  # Search for raw chunks (retrieval only)
  python run.py search "What exercises help with pain?"
  
  # Ask clinical question with AI response (full RAG)
  python run.py ask "What exercises help with Achilles tendon pain?"
  
  # Check database status
  python run.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Process PDF into embeddings')
    ingest_parser.add_argument('pdf_path', help='Path to PDF file to process')
    ingest_parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size in characters (default: 1000)')
    ingest_parser.add_argument('--overlap', type=int, default=200, help='Overlap between chunks (default: 200)')
    ingest_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Search command (retrieval only)
    search_parser = subparsers.add_parser('search', help='Search for information (retrieval only)')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return (default: 5)')
    search_parser.add_argument('--threshold', type=float, default=0.3, help='Minimum similarity threshold (default: 0.3)')
    search_parser.add_argument('--section', help='Filter by section type (treatment, diagnosis, etc.)')
    search_parser.add_argument('--no-enhance', action='store_true', help='Disable query enhancement')
    search_parser.set_defaults(func=cmd_search)
    
    # Ask command (full RAG)
    ask_parser = subparsers.add_parser('ask', help='Ask clinical question with AI response (RAG)')
    ask_parser.add_argument('query', help='Clinical question')
    ask_parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve (default: 5)')
    ask_parser.add_argument('--threshold', type=float, default=0.3, help='Minimum similarity threshold (default: 0.3)')
    ask_parser.add_argument('--section', help='Filter by section type (treatment, diagnosis, etc.)')
    ask_parser.add_argument('--no-enhance', action='store_true', help='Disable query enhancement')
    ask_parser.set_defaults(func=cmd_ask)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check database status')
    status_parser.set_defaults(func=cmd_status)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()