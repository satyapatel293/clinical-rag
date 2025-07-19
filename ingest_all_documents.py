#!/usr/bin/env python3
"""
Batch ingestion script to process all PDF documents in data/raw/ directory.
"""

import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any

# Set environment variable for ZenML on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from pipelines.document_ingestion_pipeline import pdf_processing_pipeline
from utils.database import DatabaseManager
from utils.config import DATA_DIR


def find_pdf_files(directory: str) -> List[str]:
    """Find all PDF files in the specified directory."""
    pdf_pattern = os.path.join(directory, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    return sorted(pdf_files)


def process_single_pdf(pdf_path: str, chunk_size: int = 1000, overlap: int = 200, 
                      model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Process a single PDF file using the existing pipeline."""
    print(f"📄 Processing: {Path(pdf_path).name}")
    
    try:
        pipeline_run = pdf_processing_pipeline(
            pdf_path=str(pdf_path),
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name
        )
        
        # Get the result from the storage step
        step_output = pipeline_run.steps["store_embeddings_in_database"].output.load()
        
        if step_output['success']:
            return {
                'success': True,
                'filename': Path(pdf_path).name,
                'stored_count': step_output['stored_count'],
                'error': None
            }
        else:
            return {
                'success': False,
                'filename': Path(pdf_path).name,
                'stored_count': 0,
                'error': step_output['error']
            }
            
    except Exception as e:
        return {
            'success': False,
            'filename': Path(pdf_path).name,
            'stored_count': 0,
            'error': str(e)
        }


def main():
    """Main function to process all PDFs in data/raw/ directory."""
    print("🏥 Clinical RAG - Batch Document Ingestion")
    print("=" * 50)
    
    # Find all PDF files
    raw_data_dir = os.path.join(DATA_DIR)
    pdf_files = find_pdf_files(raw_data_dir)
    
    if not pdf_files:
        print(f"❌ No PDF files found in {raw_data_dir}")
        sys.exit(1)
    
    print(f"📚 Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    results = []
    successful_count = 0
    total_chunks = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {Path(pdf_path).name}...")
        
        result = process_single_pdf(pdf_path)
        results.append(result)
        
        if result['success']:
            successful_count += 1
            total_chunks += result['stored_count']
            print(f"✅ Success: {result['stored_count']} chunks stored")
        else:
            print(f"❌ Failed: {result['error']}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {len(pdf_files)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(pdf_files) - successful_count}")
    print(f"Total chunks stored: {total_chunks}")
    
    if successful_count > 0:
        print(f"Average chunks per document: {total_chunks / successful_count:.1f}")
    
    # Show failed files
    failed_files = [r for r in results if not r['success']]
    if failed_files:
        print("\n❌ Failed files:")
        for failed in failed_files:
            print(f"  - {failed['filename']}: {failed['error']}")
    
    # Check final database status
    print("\n📊 Final Database Status:")
    try:
        db_manager = DatabaseManager()
        stats = db_manager.get_statistics()
        print(f"Documents: {stats['documents']['total_docs']} total, {stats['documents']['completed_docs']} completed")
        print(f"Chunks: {stats['chunks']['total_chunks']}")
        
        if stats['embeddings']:
            for embedding_stat in stats['embeddings']:
                print(f"Embeddings ({embedding_stat['model_name']}): {embedding_stat['total_embeddings']}")
        else:
            print("Embeddings: 0")
            
    except Exception as e:
        print(f"❌ Error checking database status: {e}")
    
    print("\n✅ Batch ingestion completed!")


if __name__ == "__main__":
    main()