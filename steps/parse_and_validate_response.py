"""Parse and validate response step for Clinical RAG generation pipeline."""

from typing import Dict, Any, List
import re
from zenml import step


#@step
def parse_and_validate_response(
    generation_result: Dict[str, Any],
    min_length: int = 50,
    max_length: int = 10000,
    require_structure: bool = True
) -> Dict[str, Any]:
    """
    Parse and validate the generated clinical response.
    
    Args:
        generation_result: Output from generate_with_ollama step
        min_length: Minimum acceptable response length
        max_length: Maximum acceptable response length  
        require_structure: Whether to require structured sections
        
    Returns:
        Dictionary containing parsed response and validation results
    """
    try:
        if not generation_result['success']:
            return {
                'parsed_response': {},
                'validation_results': {},
                'success': False,
                'error': f"Cannot parse - generation failed: {generation_result['error']}"
            }
        
        generated_text = generation_result['generated_response']
        
        # Basic validation checks
        validation_results = {
            'length_check': {
                'passed': min_length <= len(generated_text) <= max_length,
                'actual_length': len(generated_text),
                'min_required': min_length,
                'max_allowed': max_length
            },
            'content_checks': {},
            'structure_checks': {},
            'clinical_checks': {}
        }
        
        # Content validation
        validation_results['content_checks'] = {
            'has_content': len(generated_text.strip()) > 0,
            'not_empty_lines_only': len(generated_text.strip().replace('\n', '')) > 0,
            'reasonable_word_count': 10 <= len(generated_text.split()) <= 2000,
            'contains_clinical_terms': any(term in generated_text.lower() for term in [
                'clinical', 'treatment', 'patient', 'therapy', 'exercise', 'assessment', 
                'diagnosis', 'recommendation', 'evidence', 'guideline'
            ])
        }
        
        # Structure validation (if required)
        if require_structure:
            structure_patterns = {
                'has_assessment': bool(re.search(r'\b(assessment|evaluation)\b', generated_text, re.IGNORECASE)),
                'has_recommendations': bool(re.search(r'\b(recommendation|suggest|should|recommend)\b', generated_text, re.IGNORECASE)),
                'has_considerations': bool(re.search(r'\b(consider|contraindication|caution|important)\b', generated_text, re.IGNORECASE)),
                'has_sections': bool(re.search(r'\*\*.*?\*\*|^\d+\.|^[A-Z][a-z]+:', generated_text, re.MULTILINE))
            }
            validation_results['structure_checks'] = structure_patterns
        
        # Clinical content validation
        clinical_checks = {
            'mentions_evidence': bool(re.search(r'\b(evidence|research|study|literature)\b', generated_text, re.IGNORECASE)),
            'professional_tone': not bool(re.search(r'\b(i think|i believe|maybe|probably)\b', generated_text, re.IGNORECASE)),
            'includes_disclaimer': bool(re.search(r'\b(consult|healthcare professional|medical advice)\b', generated_text, re.IGNORECASE)),
            'avoids_definitive_diagnosis': not bool(re.search(r'\byou have\b|\bis diagnosed\b', generated_text, re.IGNORECASE))
        }
        validation_results['clinical_checks'] = clinical_checks
        
        # Parse response structure
        parsed_sections = _extract_sections(generated_text)
        
        # Calculate overall quality score
        all_checks = []
        all_checks.extend(validation_results['content_checks'].values())
        if require_structure:
            all_checks.extend(validation_results['structure_checks'].values())
        all_checks.extend(validation_results['clinical_checks'].values())
        
        quality_score = sum(all_checks) / len(all_checks) if all_checks else 0.0
        
        # Determine if response passes validation
        validation_passed = (
            validation_results['length_check']['passed'] and
            all(validation_results['content_checks'].values()) and
            quality_score >= 0.7  # At least 70% of checks must pass
        )
        
        parsed_response = {
            'text': generated_text,
            'sections': parsed_sections,
            'word_count': len(generated_text.split()),
            'character_count': len(generated_text),
            'quality_score': quality_score,
            'validation_passed': validation_passed
        }
        
        return {
            'parsed_response': parsed_response,
            'validation_results': validation_results,
            'metadata': {
                'validation_config': {
                    'min_length': min_length,
                    'max_length': max_length,
                    'require_structure': require_structure
                },
                'quality_metrics': {
                    'overall_score': quality_score,
                    'validation_passed': validation_passed,
                    'checks_passed': sum(all_checks),
                    'total_checks': len(all_checks)
                },
                'generation_metadata': generation_result.get('metadata', {})
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'parsed_response': {},
            'validation_results': {},
            'metadata': {
                'validation_config': {
                    'min_length': min_length,
                    'max_length': max_length,
                    'require_structure': require_structure
                }
            },
            'success': False,
            'error': f"Response parsing failed: {str(e)}"
        }


def _extract_sections(text: str) -> Dict[str, str]:
    """Extract structured sections from the response text."""
    sections = {}
    
    # Common section patterns
    section_patterns = [
        (r'\*\*Assessment\*\*\s*\n(.*?)(?=\*\*|\n\n|\Z)', 'assessment'),
        (r'\*\*Recommendations?\*\*\s*\n(.*?)(?=\*\*|\n\n|\Z)', 'recommendations'),
        (r'\*\*Considerations?\*\*\s*\n(.*?)(?=\*\*|\n\n|\Z)', 'considerations'),
        (r'Assessment\s*:?\s*\n(.*?)(?=\n[A-Z]|\n\n|\Z)', 'assessment'),
        (r'Recommendations?\s*:?\s*\n(.*?)(?=\n[A-Z]|\n\n|\Z)', 'recommendations'),
        (r'Considerations?\s*:?\s*\n(.*?)(?=\n[A-Z]|\n\n|\Z)', 'considerations')
    ]
    
    for pattern, section_name in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match and section_name not in sections:
            sections[section_name] = match.group(1).strip()
    
    # If no structured sections found, treat entire text as main content
    if not sections:
        sections['main_content'] = text.strip()
    
    return sections


#@step
def validate_clinical_content(
    parsed_result: Dict[str, Any],
    required_elements: List[str] = None
) -> Dict[str, Any]:
    """
    Additional clinical content validation step.
    
    Args:
        parsed_result: Output from parse_and_validate_response step
        required_elements: List of required clinical elements
        
    Returns:
        Dictionary with detailed clinical validation results
    """
    try:
        if not parsed_result['success']:
            return {
                'clinical_validation': {},
                'success': False,
                'error': f"Cannot validate - parsing failed: {parsed_result['error']}"
            }
        
        if required_elements is None:
            required_elements = ['evidence_based', 'professional_language', 'appropriate_disclaimers']
        
        response_text = parsed_result['parsed_response']['text']
        
        clinical_validation = {
            'evidence_based': bool(re.search(r'\b(evidence|research|clinical guidelines|studies)\b', response_text, re.IGNORECASE)),
            'professional_language': not bool(re.search(r'\b(um|uh|like|you know|basically)\b', response_text, re.IGNORECASE)),
            'appropriate_disclaimers': bool(re.search(r'\b(consult.*healthcare|medical professional|clinical guidelines)\b', response_text, re.IGNORECASE)),
            'specific_recommendations': bool(re.search(r'\b(should|recommend|consider|may benefit)\b', response_text, re.IGNORECASE)),
            'avoids_medical_diagnosis': not bool(re.search(r'\b(you have|diagnosed with|definitely)\b', response_text, re.IGNORECASE))
        }
        
        # Check for required elements
        requirements_met = all(
            clinical_validation.get(element, False) 
            for element in required_elements
        )
        
        return {
            'clinical_validation': clinical_validation,
            'requirements_met': requirements_met,
            'metadata': {
                'required_elements': required_elements,
                'validation_score': sum(clinical_validation.values()) / len(clinical_validation),
                'parsed_metadata': parsed_result.get('metadata', {})
            },
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'clinical_validation': {},
            'requirements_met': False,
            'metadata': {
                'required_elements': required_elements or []
            },
            'success': False,
            'error': f"Clinical validation failed: {str(e)}"
        }