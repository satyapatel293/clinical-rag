"""Step to load evaluation dataset for retrieval testing."""

import json
from typing import Dict, Any, List, Annotated
from zenml import step


@step
def load_evaluation_dataset(
    dataset_path: str = "evaluation/test_datasets/clinical_qa_pairs.json"
) -> Annotated[Dict[str, Any], "evaluation_dataset"]:
    """
    Load evaluation dataset from JSON file.
    
    Args:
        dataset_path: Path to evaluation dataset JSON file
        
    Returns:
        Dictionary containing evaluation pairs and metadata
    """
    try:
        with open(dataset_path, 'r') as f:
            evaluation_pairs = json.load(f)
        
        # Validate dataset structure
        if not evaluation_pairs:
            raise ValueError("Evaluation dataset is empty")
        
        # Check that all required fields are present
        required_fields = ['id', 'question', 'expected_chunk_id', 'section_type', 'filename']
        for pair in evaluation_pairs:
            for field in required_fields:
                if field not in pair:
                    raise ValueError(f"Missing required field '{field}' in evaluation pair {pair.get('id', 'unknown')}")
        
        # Group by section and document for analysis
        section_dist = {}
        doc_dist = {}
        for pair in evaluation_pairs:
            section = pair['section_type']
            doc = pair['filename']
            section_dist[section] = section_dist.get(section, 0) + 1
            doc_dist[doc] = doc_dist.get(doc, 0) + 1
        
        dataset_info = {
            'evaluation_pairs': evaluation_pairs,
            'total_pairs': len(evaluation_pairs),
            'section_distribution': section_dist,
            'document_distribution': doc_dist,
            'dataset_path': dataset_path
        }
        
        return {
            'success': True,
            'dataset_info': dataset_info,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'dataset_info': None,
            'error': str(e)
        }