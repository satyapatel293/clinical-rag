"""Test script to run the PDF extraction pipeline."""

import os
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.clinical_rag_pipeline import pdf_processing_pipeline
from utils.config import DATA_DIR

def main():
    """Test the PDF extraction pipeline with sample documents."""
    
    # Set the environment variable for ZenML
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    
    # Get all PDF files from data directory
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process the Achilles PDF (smaller document)
    test_pdf = [pdf for pdf in pdf_files if 'Achilles' in pdf.name][0] if any('Achilles' in pdf.name for pdf in pdf_files) else pdf_files[0]
    print(f"\nTesting with: {test_pdf.name}")
    
    try:
        # Run the pipeline with chunking
        pipeline_run = pdf_processing_pipeline(str(test_pdf), chunk_size=800, overlap=150)
        
        # Get the result from the embeddings step
        step_output = pipeline_run.steps["generate_embeddings"].output.load()
        
        # Display results
        if step_output['success']:
            print(f"✅ Successfully processed {test_pdf.name}")
            print(f"📄 Pages: {step_output['metadata']['num_pages']}")
            print(f"📦 Total chunks: {step_output['metadata']['total_chunks']}")
            print(f"📏 Chunk size: {step_output['metadata']['chunk_size']} chars")
            print(f"🔗 Overlap: {step_output['metadata']['overlap']} chars")
            
            # Show preprocessing info
            if step_output['metadata'].get('preprocessing_applied'):
                orig_len = step_output['metadata'].get('original_length', 0)
                clean_len = step_output['metadata'].get('cleaned_length', 0)
                print(f"🧹 Text cleaning: {orig_len:,} → {clean_len:,} chars ({((orig_len-clean_len)/orig_len*100):.1f}% reduction)")
            
            # Show embedding info
            if step_output['metadata'].get('embedding_model'):
                print(f"🤖 Embedding model: {step_output['metadata']['embedding_model']}")
                print(f"📊 Embedding dimension: {step_output['metadata']['embedding_dim']}")
                print(f"🔢 Total embeddings: {step_output['metadata']['total_embeddings']}")
            
            # Show temp file locations
            temp_files = []
            for key in ['temp_file', 'processed_temp_file', 'chunks_temp_file', 'embeddings_temp_file']:
                if key in step_output['metadata']:
                    temp_files.append(step_output['metadata'][key])
            
            if temp_files:
                print(f"\n📁 Temp files saved:")
                for file_path in temp_files:
                    print(f"   - {file_path}")
            
            # Show first few embeddings
            embeddings = step_output['embeddings'][:3]  # First 3 embeddings
            print(f"\n📖 First {len(embeddings)} embeddings:")
            print("=" * 60)
            
            for i, embedding in enumerate(embeddings):
                print(f"\n📝 Embedding {i+1} ({embedding['section_type']}):")
                print(f"   Text: {embedding['text'][:200]}..." if len(embedding['text']) > 200 else embedding['text'])
                print(f"   Text length: {len(embedding['text'])} chars")
                print(f"   Embedding shape: {embedding['embedding'].shape}")
                print(f"   Embedding preview: [{embedding['embedding'][:3]}, ..., {embedding['embedding'][-3:]}]")
                print("-" * 40)
                
        else:
            print(f"❌ Failed to process: {step_output['error']}")
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()