"""Step to generate comprehensive evaluation report and save as ZenML metadata."""

import json
import os
from datetime import datetime
from typing import Dict, Any, Annotated
from zenml import step
from zenml.logger import get_logger
from zenml import get_step_context

logger = get_logger(__name__)


@step
def generate_evaluation_report(
    metrics_result: Dict[str, Any],
    output_dir: str = "evaluation/reports",
    save_detailed: bool = True
) -> Annotated[Dict[str, Any], "evaluation_report"]:
    """
    Generate comprehensive evaluation report and save as ZenML metadata.
    
    Args:
        metrics_result: Result from calculate_evaluation_metrics step
        output_dir: Directory to save reports (optional)
        save_detailed: Whether to save detailed files
        
    Returns:
        Dictionary containing report summary and metadata
    """
    if not metrics_result['success']:
        return {
            'success': False,
            'error': f"Metrics calculation failed: {metrics_result['error']}",
            'report_summary': None
        }
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        metrics = metrics_result['metrics']
        failure_analysis = metrics_result['failure_analysis']
        parameters = metrics_result['parameters']
        
        # Generate report summary for ZenML metadata
        report_summary = {
            'timestamp': timestamp,
            'evaluation_type': 'retrieval',
            'system': 'clinical_rag',
            'total_queries': metrics_result['total_queries'],
            'success_rate': metrics['success_rate'],
            'hit_rate_at_1': metrics.get('hit_rate_at_1', 0.0),
            'hit_rate_at_3': metrics.get('hit_rate_at_3', 0.0),
            'hit_rate_at_5': metrics.get('hit_rate_at_5', 0.0),
            'hit_rate_at_10': metrics.get('hit_rate_at_10', 0.0),
            'precision_at_1': metrics['precision_at_1'],
            'mean_reciprocal_rank': metrics['mean_reciprocal_rank'],
            'performance_category': metrics_result['performance_category'],
            'total_failures': failure_analysis['total_failures'],
            'failure_rate': failure_analysis['failure_rate']
        }
        
        # Add rank statistics if available
        if metrics['average_rank'] is not None:
            report_summary.update({
                'average_rank': metrics['average_rank'],
                'median_rank': metrics['median_rank'],
                'min_rank': metrics['min_rank'],
                'max_rank': metrics['max_rank']
            })
        
        # Add evaluation parameters
        report_summary['parameters'] = parameters
        
        # Add failure analysis summary
        report_summary['failure_analysis'] = {
            'total_failures': failure_analysis['total_failures'],
            'failure_rate': failure_analysis['failure_rate'],
            'top_failing_sections': dict(list(sorted(
                failure_analysis['failures_by_section'].items(), 
                key=lambda x: x[1], reverse=True
            )[:3])) if failure_analysis['failures_by_section'] else {},
            'top_failing_documents': dict(list(sorted(
                failure_analysis['failures_by_document'].items(), 
                key=lambda x: x[1], reverse=True
            )[:3])) if failure_analysis['failures_by_document'] else {}
        }
        
        # Generate performance insights
        hit_rate_5 = metrics.get('hit_rate_at_5', 0.0)
        if hit_rate_5 >= 0.7:
            recommendation = "excellent_performance"
            insights = "System ready for production use"
        elif hit_rate_5 >= 0.5:
            recommendation = "good_with_improvements"
            insights = "Consider tuning similarity threshold or embedding model"
        else:
            recommendation = "needs_improvement"
            insights = "Review query enhancement and training data"
        
        report_summary['recommendation'] = recommendation
        report_summary['insights'] = insights
        
        # Save as ZenML metadata
        try:
            step_context = get_step_context()
            
            # Log key metrics as metadata
            step_context.add_output_metadata(
                output_name="evaluation_report",
                metadata={
                    "evaluation_timestamp": timestamp,
                    "total_queries": metrics_result['total_queries'],
                    "success_rate": float(metrics['success_rate']),
                    "hit_rate_at_5": float(metrics.get('hit_rate_at_5', 0.0)),
                    "mean_reciprocal_rank": float(metrics['mean_reciprocal_rank']),
                    "performance_category": metrics_result['performance_category'],
                    "recommendation": recommendation,
                    "evaluation_parameters": parameters,
                    "failure_summary": report_summary['failure_analysis']
                }
            )
            
            logger.info(f"Evaluation metadata saved to ZenML with timestamp: {timestamp}")
            logger.info(f"Hit Rate@5: {hit_rate_5:.3f}, Performance: {metrics_result['performance_category']}")
            
        except Exception as metadata_error:
            logger.warning(f"Failed to save ZenML metadata: {metadata_error}")
        
        # Optionally save detailed files
        file_paths = {}
        if save_detailed and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save detailed JSON report
                detailed_report = {
                    'metadata': report_summary,
                    'detailed_metrics': metrics,
                    'detailed_failure_analysis': failure_analysis,
                    'all_results': metrics_result.get('results', [])
                }
                
                json_report_path = os.path.join(output_dir, f"evaluation_detailed_{timestamp}.json")
                with open(json_report_path, 'w') as f:
                    json.dump(detailed_report, f, indent=2)
                
                file_paths['detailed_json'] = json_report_path
                logger.info(f"Detailed report saved to: {json_report_path}")
                
            except Exception as file_error:
                logger.warning(f"Failed to save detailed files: {file_error}")
        
        return {
            'success': True,
            'report_summary': report_summary,
            'file_paths': file_paths,
            'zenml_metadata_saved': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'report_summary': None,
            'zenml_metadata_saved': False
        }