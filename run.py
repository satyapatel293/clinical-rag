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

from pipelines.document_ingestion_pipeline import pdf_processing_pipeline
from pipelines.response_generation_pipeline import clinical_generation_pipeline
from pipelines.simple_generation_pipeline import simple_clinical_generation_pipeline
from pipelines.evaluation_pipeline import retrieval_evaluation_pipeline
from utils.search import ClinicalRAGSearcher
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
            print("🤖 Generating simple response without context...\n")
            
            try:
                # Use simple generation pipeline without context
                pipeline_run = simple_clinical_generation_pipeline(
                    query=args.query,
                    model_name=args.model,
                    temperature=args.temperature,
                    output_format="detailed"
                )
                
                # Get the output from the last step
                pipeline_result = pipeline_run.steps["format_final_cli_output"].output.load()
                
                if pipeline_result['success']:
                    print(pipeline_result['cli_output'])
                else:
                    print(f"❌ Generation failed: {pipeline_result['error']}")
                    
            except Exception as e:
                print(f"❌ Simple generation error: {e}")
            return
            
        print(f"✅ Found {len(results)} relevant chunks")
        print("🤖 Running generation pipeline...\n")
        
        try:
            # Convert search results to format expected by pipeline
            chunks_for_generation = []
            for result in results:
                chunks_for_generation.append({
                    'text': result['text'],
                    'similarity': result['similarity'],
                    'section_type': result['section_type'],
                    'filename': result['filename']
                })
            
            # Run the clinical generation pipeline
            pipeline_run = clinical_generation_pipeline(
                query=args.query,
                retrieved_chunks=chunks_for_generation,
                model_name=args.model,
                temperature=args.temperature,
                output_format="detailed",
                include_full_metadata=True
            )
            
            # Get the output from the last step
            pipeline_result = pipeline_run.steps["format_final_cli_output"].output.load()
            
            if pipeline_result['success']:
                # Pipeline handles all formatting, just print the result
                print(pipeline_result['cli_output'])
            else:
                print(f"❌ Generation pipeline failed: {pipeline_result['error']}")
                print("\n📄 Retrieved information (fallback):")
                _show_chunks(results)
                
        except Exception as e:
            print(f"❌ Pipeline execution error: {e}")
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


def cmd_eval(args):
    """Run retrieval evaluation pipeline."""
    try:
        print("🧪 Clinical RAG - Automated Retrieval Evaluation")
        print("=" * 50)
        
        if args.comparative:
            print("🔄 Running comparative evaluation across multiple configurations...")
            print("⚠️  Note: Running separate evaluations for each configuration")
            
            # Define configurations to compare
            configurations = [
                {"top_k": 5, "similarity_threshold": 0.3, "enhance_query": True, "name": "baseline"},
                {"top_k": 10, "similarity_threshold": 0.3, "enhance_query": True, "name": "top10"},
                {"top_k": 5, "similarity_threshold": 0.2, "enhance_query": True, "name": "lower_threshold"},
                {"top_k": 5, "similarity_threshold": 0.3, "enhance_query": False, "name": "no_enhancement"}
            ]
            
            comparison_results = []
            
            # Run separate pipeline for each configuration
            for i, config in enumerate(configurations, 1):
                print(f"\n[{i}/{len(configurations)}] Testing configuration: {config['name']}")
                print(f"  Parameters: top_k={config['top_k']}, threshold={config['similarity_threshold']}, enhance={config['enhance_query']}")
                
                pipeline_run = retrieval_evaluation_pipeline(
                    dataset_path=args.dataset,
                    top_k=config["top_k"],
                    similarity_threshold=config["similarity_threshold"],
                    enhance_query=config["enhance_query"],
                    k_values=[1, 3, 5, 10],
                    save_detailed_report=False,  # Don't save individual reports for comparison
                    report_output_dir="evaluation/reports"
                )
                
                # Get the final report
                final_report = pipeline_run.steps["generate_evaluation_report"].output.load()
                
                if final_report['success']:
                    summary = final_report['report_summary']
                    summary['config_name'] = config['name']
                    summary['config'] = config
                    comparison_results.append(summary)
                    print(f"  ✅ Hit Rate@5: {summary['hit_rate_at_5']:.3f}")
                else:
                    print(f"  ❌ Failed: {final_report['error']}")
            
            # Show comparison results
            if comparison_results:
                print(f"\n✅ Comparative evaluation completed!")
                print(f"📊 Tested {len(comparison_results)} configurations")
                
                # Find best configuration
                best_config = max(comparison_results, key=lambda x: x['hit_rate_at_5'])
                print(f"\n🏆 Best configuration: {best_config['config_name']}")
                print(f"   Hit Rate@5: {best_config['hit_rate_at_5']:.3f}")
                print(f"   Parameters: {best_config['config']}")
                
                # Show all results
                print(f"\n📈 All Results:")
                print(f"{'Configuration':<15} {'Hit@5':<8} {'MRR':<8} {'Success%':<10}")
                print("-" * 45)
                for result in comparison_results:
                    print(f"{result['config_name']:<15} {result['hit_rate_at_5']:<8.3f} {result['mean_reciprocal_rank']:<8.3f} {result['success_rate']*100:<10.1f}")
            else:
                print(f"❌ All comparative evaluations failed")
                
        else:
            print(f"🔍 Running single evaluation with parameters:")
            print(f"  Dataset: {args.dataset}")
            print(f"  Top K: {args.top_k}")
            print(f"  Threshold: {args.threshold}")
            print(f"  Query enhancement: {not args.no_enhance}")
            
            pipeline_run = retrieval_evaluation_pipeline(
                dataset_path=args.dataset,
                top_k=args.top_k,
                similarity_threshold=args.threshold,
                enhance_query=not args.no_enhance,
                k_values=[1, 3, 5, 10],
                save_detailed_report=not args.no_save,
                report_output_dir="evaluation/reports"
            )
            
            # Get the final report
            final_report = pipeline_run.steps["generate_evaluation_report"].output.load()
            
            if final_report['success']:
                summary = final_report['report_summary']
                print(f"\n✅ Evaluation completed successfully!")
                print(f"📊 Results saved as ZenML metadata")
                print(f"\n🎯 Key Metrics:")
                print(f"  Success Rate: {summary['success_rate']:.3f} ({summary['success_rate']*100:.1f}%)")
                print(f"  Hit Rate@5: {summary['hit_rate_at_5']:.3f} ({summary['hit_rate_at_5']*100:.1f}%)")
                print(f"  Mean Reciprocal Rank: {summary['mean_reciprocal_rank']:.3f}")
                print(f"  Performance: {summary['performance_category'].upper()}")
                print(f"  Recommendation: {summary['recommendation']}")
                
                if final_report.get('file_paths'):
                    print(f"\n📄 Detailed reports saved:")
                    for report_type, path in final_report['file_paths'].items():
                        print(f"  {report_type}: {path}")
            else:
                print(f"❌ Evaluation failed: {final_report['error']}")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Evaluation pipeline error: {e}")
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
    ask_parser.add_argument('--model', default='llama3.2:3b', help='Ollama model for generation (default: llama3.2:3b)')
    ask_parser.add_argument('--temperature', type=float, default=0.1, help='Generation temperature 0.0-1.0 (default: 0.1)')
    ask_parser.set_defaults(func=cmd_ask)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check database status')
    status_parser.set_defaults(func=cmd_status)
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run automated retrieval evaluation')
    eval_parser.add_argument('--dataset', default='evaluation/test_datasets/clinical_qa_pairs.json', 
                            help='Path to evaluation dataset (default: evaluation/test_datasets/clinical_qa_pairs.json)')
    eval_parser.add_argument('--top-k', type=int, default=10, 
                            help='Number of top results to retrieve (default: 10)')
    eval_parser.add_argument('--threshold', type=float, default=0.3, 
                            help='Similarity threshold (default: 0.3)')
    eval_parser.add_argument('--no-enhance', action='store_true', 
                            help='Disable query enhancement')
    eval_parser.add_argument('--comparative', action='store_true', 
                            help='Run comparative evaluation across multiple configurations')
    eval_parser.add_argument('--no-save', action='store_true', 
                            help='Do not save detailed reports to disk')
    eval_parser.set_defaults(func=cmd_eval)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()