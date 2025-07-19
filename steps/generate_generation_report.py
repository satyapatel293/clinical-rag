"""Step to generate comprehensive generation evaluation report and save as ZenML metadata."""

import json
import os
from datetime import datetime
from typing import Dict, Any, Annotated
from zenml import step
from zenml.logger import get_logger
from zenml import get_step_context

logger = get_logger(__name__)


@step
def generate_generation_report(
    metrics_result: Dict[str, Any],
    output_dir: str = "evaluation/reports/generation",
    save_detailed: bool = True
) -> Annotated[Dict[str, Any], "generation_evaluation_report"]:
    """
    Generate comprehensive generation evaluation report and save as ZenML metadata.
    
    Args:
        metrics_result: Result from calculate_generation_metrics step
        output_dir: Directory to save reports
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
        
        # Generate report summary
        report_summary = {
            'timestamp': timestamp,
            'evaluation_type': 'generation_llm_judge',
            'system': 'clinical_rag',
            'total_responses': metrics['total_responses'],
            'successful_evaluations': metrics['successful_evaluations'],
            'overall_score': f"{metrics['overall_score']:.1f}/7",
            'overall_percentage': f"{metrics['overall_percentage']:.1f}%",
            'pass_rate': f"{metrics['pass_rate']:.1%}",
            'performance_category': metrics['performance_category'],
            'generation_model': parameters['generation_model'],
            'judge_model': parameters['judge_model']
        }
        
        # Add dimension breakdowns
        for dimension, dim_metrics in metrics['dimension_metrics'].items():
            report_summary[f'{dimension}_score'] = f"{dim_metrics['average_score']:.1f}/{dim_metrics['max_possible']}"
            report_summary[f'{dimension}_percentage'] = f"{dim_metrics['percentage']:.1f}%"
        
        # Add evaluation parameters
        report_summary['parameters'] = parameters
        
        # Add failure analysis summary
        report_summary['failure_analysis'] = {
            'total_failures': failure_analysis['total_failures'],
            'failure_rate': f"{failure_analysis['failure_rate']:.1%}",
            'top_failing_sections': dict(list(sorted(
                failure_analysis['failures_by_section'].items(), 
                key=lambda x: x[1], reverse=True
            )[:3])) if failure_analysis['failures_by_section'] else {},
            'top_failing_documents': dict(list(sorted(
                failure_analysis['failures_by_document'].items(), 
                key=lambda x: x[1], reverse=True
            )[:3])) if failure_analysis['failures_by_document'] else {}
        }
        
        # Generate insights and recommendations
        if metrics['overall_percentage'] >= 80:
            recommendation = "excellent_performance"
            insights = "System ready for production use with high-quality generations"
        elif metrics['overall_percentage'] >= 70:
            recommendation = "good_with_minor_improvements"
            insights = "Strong performance, consider fine-tuning for specific clinical domains"
        elif metrics['overall_percentage'] >= 60:
            recommendation = "fair_needs_optimization" 
            insights = "Moderate performance, review generation prompts and training data"
        else:
            recommendation = "needs_significant_improvement"
            insights = "Poor performance, consider different models or major prompt revisions"
        
        report_summary['recommendation'] = recommendation
        report_summary['insights'] = insights
        
        # Save as ZenML metadata
        zenml_metadata_saved = False
        try:
            step_context = get_step_context()
            
            # Log comprehensive metadata
            step_context.add_output_metadata(
                output_name="generation_evaluation_report",
                metadata={
                    "evaluation_timestamp": timestamp,
                    "evaluation_type": "generation_llm_judge",
                    "total_responses": int(metrics['total_responses']),
                    "successful_evaluations": int(metrics['successful_evaluations']),
                    "overall_score": float(metrics['overall_score']),
                    "overall_percentage": float(metrics['overall_percentage']),
                    "pass_rate": float(metrics['pass_rate']),
                    "performance_category": metrics['performance_category'],
                    "generation_model": parameters['generation_model'],
                    "judge_model": parameters['judge_model'],
                    "recommendation": recommendation,
                    "insights": insights,
                    "evaluation_parameters": parameters,
                    "failure_summary": report_summary['failure_analysis'],
                    "dimension_scores": {
                        dim: float(scores['percentage']) 
                        for dim, scores in metrics['dimension_metrics'].items()
                    }
                }
            )
            
            zenml_metadata_saved = True
            logger.info(f"Generation evaluation metadata saved to ZenML with timestamp: {timestamp}")
            logger.info(f"Overall Score: {metrics['overall_score']:.1f}/7 ({metrics['overall_percentage']:.1f}%)")
            logger.info(f"Performance Category: {metrics['performance_category']}")
            
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
                    'all_results': metrics_result.get('all_results', [])
                }
                
                json_report_path = os.path.join(output_dir, f"generation_evaluation_{timestamp}.json")
                with open(json_report_path, 'w') as f:
                    json.dump(detailed_report, f, indent=2)
                
                file_paths['detailed_json'] = json_report_path
                logger.info(f"Detailed generation report saved to: {json_report_path}")
                
                # Save summary report for easy reading
                summary_path = os.path.join(output_dir, f"generation_summary_{timestamp}.txt")
                with open(summary_path, 'w') as f:
                    f.write("📊 Clinical RAG Generation Evaluation Summary\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Generation Model: {parameters['generation_model']}\n")
                    f.write(f"Judge Model: {parameters['judge_model']}\n\n")
                    f.write(f"Overall Score: {report_summary['overall_score']} ({report_summary['overall_percentage']})\n")
                    f.write(f"Pass Rate: {report_summary['pass_rate']}\n")
                    f.write(f"Performance Category: {metrics['performance_category']}\n\n")
                    f.write("Dimension Scores:\n")
                    for dimension, dim_metrics in metrics['dimension_metrics'].items():
                        f.write(f"  {dimension.replace('_', ' ').title()}: {dim_metrics['average_score']:.1f}/{dim_metrics['max_possible']} ({dim_metrics['percentage']:.1f}%)\n")
                    f.write(f"\nRecommendation: {recommendation}\n")
                    f.write(f"Insights: {insights}\n")
                
                file_paths['summary_txt'] = summary_path
                
            except Exception as file_error:
                logger.warning(f"Failed to save detailed files: {file_error}")
        
        return {
            'success': True,
            'report_summary': report_summary,
            'file_paths': file_paths,
            'zenml_metadata_saved': zenml_metadata_saved,
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