"""Test script to run the PDF extraction pipeline."""

import os
from pathlib import Path
from src.pipeline import pdf_processing_pipeline
from src.config import DATA_DIR

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
    
    # Process the first PDF as a test
    test_pdf = pdf_files[0]
    print(f"\nTesting with: {test_pdf.name}")
    
    try:
        # Run the pipeline with chunking
        pipeline_run = pdf_processing_pipeline(str(test_pdf), chunk_size=800, overlap=150)
        
        # Get the result from the chunking step
        step_output = pipeline_run.steps["chunk_text"].output.load()
        
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
            
            # Show temp file locations
            temp_files = []
            for key in ['temp_file', 'processed_temp_file', 'chunks_temp_file']:
                if key in step_output['metadata']:
                    temp_files.append(step_output['metadata'][key])
            
            if temp_files:
                print(f"\n📁 Temp files saved:")
                for file_path in temp_files:
                    print(f"   - {file_path}")
            
            # Show first few chunks
            chunks = step_output['chunks'][:3]  # First 3 chunks
            print(f"\n📖 First {len(chunks)} chunks:")
            print("=" * 60)
            
            for i, chunk in enumerate(chunks):
                print(f"\n📝 Chunk {i+1} ({chunk['section_type']}):")
                print(f"   Text: {chunk['text'][:200]}..." if len(chunk['text']) > 200 else chunk['text'])
                print(f"   Length: {len(chunk['text'])} chars")
                print("-" * 40)
                
        else:
            print(f"❌ Failed to process: {step_output['error']}")
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()