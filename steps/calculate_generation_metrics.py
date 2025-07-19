"""Step to calculate aggregate metrics from LLM judge evaluation results."""

from typing import Dict, Any, Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def calculate_generation_metrics(
    evaluation_results: Dict[str, Any]
) -> Annotated[Dict[str, Any], "generation_metrics"]:
    """
    Calculate aggregate metrics from LLM judge evaluation results.
    
    Args:
        evaluation_results: Results from generate_and_evaluate_responses step
        
    Returns:
        Dictionary containing calculated metrics and analysis
    """
    if not evaluation_results['success']:
        return {
            'success': False,
            'error': f"Evaluation failed: {evaluation_results['error']}",
            'metrics': None
        }
    
    try:
        results = evaluation_results['results']
        parameters = evaluation_results['parameters']
        
        # Filter successful evaluations
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                'success': False,
                'error': "No successful evaluations to calculate metrics from",
                'metrics': None
            }
        
        # Calculate overall metrics
        total_responses = len(results)
        successful_evaluations = len(successful_results)
        
        # Score metrics
        scores = [r['judge_score'] for r in successful_results]
        percentages = [r['judge_percentage'] for r in successful_results]
        
        overall_score = sum(scores) / len(scores) if scores else 0
        overall_percentage = sum(percentages) / len(percentages) if percentages else 0
        
        # Pass rate (consider 5/7 or higher as "passing")
        pass_threshold = 5
        passing_responses = [r for r in successful_results if r['judge_score'] >= pass_threshold]
        pass_rate = len(passing_responses) / len(successful_results) if successful_results else 0
        
        # Dimension-specific metrics
        dimension_metrics = {}
        for dimension in ['clinical_accuracy', 'relevance', 'evidence_support']:
            dimension_scores = []
            for r in successful_results:
                if 'dimension_scores' in r and dimension in r['dimension_scores']:
                    dimension_scores.append(r['dimension_scores'][dimension])
            
            if dimension_scores:
                max_possible = 3 if dimension == 'clinical_accuracy' else 2  # Questions per dimension
                dimension_metrics[dimension] = {
                    'average_score': sum(dimension_scores) / len(dimension_scores),
                    'max_possible': max_possible,
                    'percentage': (sum(dimension_scores) / len(dimension_scores) / max_possible) * 100
                }
        
        # Failure analysis
        failed_results = [r for r in results if not r['success'] or r['judge_score'] < pass_threshold]
        
        # Group failures by section type
        section_failures = {}
        for failure in failed_results:
            section = failure.get('section_type', 'unknown')
            section_failures[section] = section_failures.get(section, 0) + 1
        
        # Group failures by document
        doc_failures = {}
        for failure in failed_results:
            doc = failure.get('filename', 'unknown')
            doc_failures[doc] = doc_failures.get(doc, 0) + 1
        
        failure_analysis = {
            'total_failures': len(failed_results),
            'failure_rate': len(failed_results) / total_responses if total_responses else 0,
            'failures_by_section': section_failures,
            'failures_by_document': doc_failures,
            'low_scoring_examples': sorted(failed_results, key=lambda x: x.get('judge_score', 0))[:5]
        }
        
        # Performance categorization
        if overall_percentage >= 80:
            performance_category = "excellent"
        elif overall_percentage >= 70:
            performance_category = "good"
        elif overall_percentage >= 60:
            performance_category = "fair"
        else:
            performance_category = "needs_improvement"
        
        metrics = {
            'total_responses': total_responses,
            'successful_evaluations': successful_evaluations,
            'evaluation_success_rate': successful_evaluations / total_responses,
            'overall_score': overall_score,
            'overall_percentage': overall_percentage,
            'pass_rate': pass_rate,
            'pass_threshold': pass_threshold,
            'dimension_metrics': dimension_metrics,
            'performance_category': performance_category,
            'score_distribution': {
                'min_score': min(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'median_score': sorted(scores)[len(scores)//2] if scores else 0
            }
        }
        
        logger.info(f"Generation metrics calculated: {overall_score:.1f}/7 ({overall_percentage:.1f}%)")
        
        return {
            'success': True,
            'metrics': metrics,
            'failure_analysis': failure_analysis,
            'parameters': parameters,
            'total_evaluations': total_responses,
            'all_results': results,  # Include all results for detailed reporting
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'metrics': None
        }