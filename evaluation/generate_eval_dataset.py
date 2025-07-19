#!/usr/bin/env python3
"""
Generate evaluation dataset by sampling chunks and creating PT-realistic questions.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any

# Set environment variable for ZenML on macOS
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from utils.database import DatabaseManager
from litellm import completion


def sample_random_chunks(db_manager: DatabaseManager, 
                        num_chunks: int = 300) -> List[Dict[str, Any]]:
    """
    Sample random chunks from the database for manual review.
    
    Selects chunks that are:
    - Random across all documents
    - Substantial in content (>100 characters)
    - Include variety for manual selection
    """
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            # Get random chunks with basic filtering
            query = """
            SELECT 
                c.id,
                c.text,
                c.section_type,
                d.filename,
                c.char_length
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.char_length > 100
              AND c.text IS NOT NULL
              AND c.text != ''
            ORDER BY RANDOM()
            LIMIT %s;
            """
            
            cur.execute(query, (num_chunks,))
            results = cur.fetchall()
            
            # Convert to list of dictionaries
            chunks = []
            for row in results:
                chunks.append({
                    'chunk_id': row[0],
                    'text': row[1],
                    'section_type': row[2] or 'general',
                    'filename': row[3],
                    'char_length': row[4],
                    'selected': False  # For manual review
                })
            
            return chunks


def generate_pt_question(chunk_text: str, section_type: str, filename: str) -> str:
    """
    Generate a physical therapist-realistic question for a given chunk.
    """
    # Create context-aware prompt based on section type
    section_context = {
        'treatment': 'treatment interventions, therapeutic techniques, or clinical procedures',
        'diagnosis': 'diagnostic criteria, assessment methods, or clinical findings',
        'exercises': 'therapeutic exercises, rehabilitation protocols, or movement patterns',
        'prevention': 'injury prevention strategies, risk factors, or preventive measures',
        'general': 'clinical information, patient care, or therapeutic approaches'
    }
    
    context = section_context.get(section_type, section_context['general'])
    
    # Extract clinical domain from filename
    clinical_domain = filename.replace('.pdf', '').replace('_', ' ').title()
    
    prompt = f"""You are a physical therapist working in a clinical setting. Based on the following clinical text about {clinical_domain}, generate a realistic question that a physical therapist would ask when looking for this specific information.

The question should be:
- Natural and conversational (how a PT would actually ask)
- Specific enough to lead to this exact information
- Focused on {context}
- Practical for clinical decision-making

Clinical text:
{chunk_text}

Generate only the question, no additional text:"""

    try:
        response = completion(
            model="ollama/llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        question = response.choices[0].message.content.strip()
        
        # Clean up the question
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
            
        return question
        
    except Exception as e:
        print(f"Error generating question: {e}")
        # Fallback to a generic question
        return f"What is the recommended approach for {clinical_domain.lower()}?"


def generate_candidate_chunks(output_file: str = "evaluation/test_datasets/candidate_chunks.json",
                             num_chunks: int = 300):
    """
    Generate candidate chunks for manual review.
    """
    print(f"🔍 Generating {num_chunks} candidate chunks for manual review...")
    
    # Sample random chunks from database
    print("📊 Sampling random chunks from database...")
    db_manager = DatabaseManager()
    chunks = sample_random_chunks(db_manager, num_chunks)
    
    print(f"✅ Selected {len(chunks)} random chunks from database")
    
    # Show distribution
    doc_dist = {}
    section_dist = {}
    for chunk in chunks:
        doc_dist[chunk['filename']] = doc_dist.get(chunk['filename'], 0) + 1
        section_dist[chunk['section_type']] = section_dist.get(chunk['section_type'], 0) + 1
    
    print(f"📄 Document distribution: {len(doc_dist)} documents")
    print(f"📋 Section distribution: {dict(section_dist)}")
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✅ Candidate chunks saved to {output_file}")
    print(f"📝 Please review and mark 'selected': true for chunks you want to use")
    print(f"   Then run with --generate-questions to create the final dataset")
    
    return chunks


def generate_questions_from_selected(candidate_file: str = "evaluation/test_datasets/candidate_chunks.json",
                                   output_file: str = "evaluation/test_datasets/clinical_qa_pairs.json"):
    """
    Generate questions for manually selected chunks.
    """
    print("🔍 Loading candidate chunks...")
    
    # Load candidate chunks
    with open(candidate_file, 'r') as f:
        candidate_chunks = json.load(f)
    
    # Filter to selected chunks only
    selected_chunks = [chunk for chunk in candidate_chunks if chunk.get('selected', False)]
    
    if not selected_chunks:
        print("❌ No chunks marked as selected. Please edit the candidate file and mark chunks with 'selected': true")
        return []
    
    print(f"✅ Found {len(selected_chunks)} selected chunks out of {len(candidate_chunks)} candidates")
    
    # Generate questions for selected chunks
    print("\n🤖 Generating PT-realistic questions for selected chunks...")
    evaluation_pairs = []
    
    for i, chunk in enumerate(selected_chunks, 1):
        print(f"[{i}/{len(selected_chunks)}] Generating question for chunk {chunk['chunk_id']}...")
        
        question = generate_pt_question(
            chunk['text'], 
            chunk['section_type'], 
            chunk['filename']
        )
        
        evaluation_pair = {
            'id': f"eval_{i:03d}",
            'question': question,
            'expected_chunk_id': chunk['chunk_id'],
            'chunk_text': chunk['text'],
            'section_type': chunk['section_type'],
            'filename': chunk['filename'],
            'char_length': chunk['char_length']
        }
        
        evaluation_pairs.append(evaluation_pair)
        
        # Show a few examples
        if i <= 3:
            print(f"  Q: {question}")
            print(f"  A: {chunk['text'][:100]}...")
            print()
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(evaluation_pairs, f, indent=2)
    
    print(f"✅ Evaluation dataset saved to {output_file}")
    print(f"📊 Generated {len(evaluation_pairs)} question-chunk pairs")
    
    return evaluation_pairs


def main():
    """Main function to generate evaluation dataset."""
    parser = argparse.ArgumentParser(description='Generate evaluation dataset for Clinical RAG')
    parser.add_argument('--generate-candidates', action='store_true',
                       help='Generate candidate chunks for manual review')
    parser.add_argument('--generate-questions', action='store_true',
                       help='Generate questions from selected chunks')
    parser.add_argument('--num-chunks', type=int, default=300,
                       help='Number of candidate chunks to generate (default: 300)')
    parser.add_argument('--candidate-file', default='evaluation/test_datasets/candidate_chunks.json',
                       help='Path to candidate chunks file')
    parser.add_argument('--output-file', default='evaluation/test_datasets/clinical_qa_pairs.json',
                       help='Path to output evaluation dataset')
    
    args = parser.parse_args()
    
    print("🏥 Clinical RAG - Evaluation Dataset Generation")
    print("=" * 50)
    
    try:
        if args.generate_candidates:
            # Phase 1: Generate candidate chunks
            chunks = generate_candidate_chunks(
                output_file=args.candidate_file,
                num_chunks=args.num_chunks
            )
            print(f"\n✅ Successfully generated {len(chunks)} candidate chunks!")
            
        elif args.generate_questions:
            # Phase 2: Generate questions from selected chunks
            dataset = generate_questions_from_selected(
                candidate_file=args.candidate_file,
                output_file=args.output_file
            )
            print(f"\n✅ Successfully generated {len(dataset)} evaluation pairs!")
            
        else:
            # Default: Generate candidate chunks
            print("No specific action specified. Generating candidate chunks...")
            chunks = generate_candidate_chunks(
                output_file=args.candidate_file,
                num_chunks=args.num_chunks
            )
            print(f"\n✅ Successfully generated {len(chunks)} candidate chunks!")
        
    except Exception as e:
        print(f"❌ Error generating evaluation dataset: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())