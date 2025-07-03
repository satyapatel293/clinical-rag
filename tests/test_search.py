#!/usr/bin/env python3
"""
Test script for Clinical RAG search functionality.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.search import ClinicalRAGSearcher

def main():
    # Initialize searcher
    print("Initializing Clinical RAG searcher...")
    searcher = ClinicalRAGSearcher()
    
    # Test queries
    test_queries = [
        "What exercises help with Achilles pain?",
        "How do you diagnose Achilles tendinopathy?", 
        "What are the risk factors for Achilles problems?",
        "Should patients rest completely with Achilles tendinopathy?",
        "What is the recommended treatment approach?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)
        
        # General search
        results = searcher.search_similar_chunks(query, top_k=3)
        searcher.print_search_results(results)
        
        # If it's a treatment question, also show treatment-specific results
        if "exercise" in query.lower() or "treatment" in query.lower() or "rest" in query.lower():
            print(f"\n--- Treatment-specific results for: {query} ---")
            treatment_results = searcher.get_treatment_recommendations(query, top_k=2)
            searcher.print_search_results(treatment_results)
        
        # If it's a diagnosis question, show diagnostic results
        elif "diagnose" in query.lower() or "risk" in query.lower():
            print(f"\n--- Diagnostic-specific results for: {query} ---")
            diagnostic_results = searcher.get_diagnostic_info(query, top_k=2)
            searcher.print_search_results(diagnostic_results)
    
    # Test section filtering
    print(f"\n{'='*60}")
    print("SECTION FILTERING TEST")
    print('='*60)
    
    sections = ["treatment", "diagnosis", "methods"]
    for section in sections:
        print(f"\n--- {section.upper()} section search ---")
        results = searcher.search_by_section("Achilles tendinopathy", section, top_k=2)
        searcher.print_search_results(results)

if __name__ == "__main__":
    main()