#!/usr/bin/env python3
"""
Utility functions for loading and processing evaluation results.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd


class EvaluationLoader:
    """Utility class for loading and processing evaluation results."""
    
    def __init__(self, reports_dir: str = "evaluation/reports"):
        """Initialize the evaluation loader.
        
        Args:
            reports_dir: Directory containing evaluation reports
        """
        self.reports_dir = reports_dir
    
    def get_available_reports(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available evaluation reports organized by type.
        
        Returns:
            Dictionary with 'retrieval' and 'generation' keys containing report metadata
        """
        reports = {
            'retrieval': [],
            'generation': []
        }
        
        # Find retrieval evaluation reports
        retrieval_pattern = os.path.join(self.reports_dir, "evaluation_detailed_*.json")
        for file_path in glob.glob(retrieval_pattern):
            try:
                metadata = self._extract_report_metadata(file_path, 'retrieval')
                if metadata:
                    reports['retrieval'].append(metadata)
            except Exception as e:
                print(f"Warning: Could not process retrieval report {file_path}: {e}")
        
        # Find generation evaluation reports
        generation_pattern = os.path.join(self.reports_dir, "generation/generation_evaluation_*.json")
        for file_path in glob.glob(generation_pattern):
            try:
                metadata = self._extract_report_metadata(file_path, 'generation')
                if metadata:
                    reports['generation'].append(metadata)
            except Exception as e:
                print(f"Warning: Could not process generation report {file_path}: {e}")
        
        # Sort by timestamp (most recent first)
        for report_type in reports:
            reports[report_type].sort(key=lambda x: x['timestamp'], reverse=True)
        
        return reports
    
    def load_report(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a specific evaluation report.
        
        Args:
            file_path: Path to the evaluation report JSON file
            
        Returns:
            Loaded report data or None if loading fails
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading report {file_path}: {e}")
            return None
    
    def get_latest_report(self, report_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest report of a specific type.
        
        Args:
            report_type: Either 'retrieval' or 'generation'
            
        Returns:
            Latest report data or None if no reports found
        """
        reports = self.get_available_reports()
        if report_type not in reports or not reports[report_type]:
            return None
        
        latest_metadata = reports[report_type][0]
        return self.load_report(latest_metadata['file_path'])
    
    def _extract_report_metadata(self, file_path: str, report_type: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from a report file.
        
        Args:
            file_path: Path to the report file
            report_type: Type of report ('retrieval' or 'generation')
            
        Returns:
            Report metadata dictionary
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            
            # Extract timestamp from filename if not in metadata
            filename = os.path.basename(file_path)
            if 'timestamp' not in metadata:
                if 'evaluation_detailed_' in filename:
                    timestamp_str = filename.replace('evaluation_detailed_', '').replace('.json', '')
                elif 'generation_evaluation_' in filename:
                    timestamp_str = filename.replace('generation_evaluation_', '').replace('.json', '')
                else:
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                metadata['timestamp'] = timestamp_str
            
            return {
                'file_path': file_path,
                'filename': filename,
                'timestamp': metadata.get('timestamp'),
                'report_type': report_type,
                'evaluation_type': metadata.get('evaluation_type', report_type),
                'total_queries': metadata.get('total_queries', data.get('parameters', {}).get('total_queries', 0)),
                'performance_summary': self._get_performance_summary(data, report_type),
                'metadata': metadata
            }
        except Exception as e:
            print(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    def _get_performance_summary(self, data: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """Extract key performance metrics for quick summary.
        
        Args:
            data: Full report data
            report_type: Type of report
            
        Returns:
            Summary metrics dictionary
        """
        if report_type == 'retrieval':
            metadata = data.get('metadata', {})
            return {
                'hit_rate_at_5': metadata.get('hit_rate_at_5', 0.0),
                'success_rate': metadata.get('success_rate', 0.0),
                'mean_reciprocal_rank': metadata.get('mean_reciprocal_rank', 0.0),
                'performance_category': metadata.get('performance_category', 'unknown'),
                'failure_rate': metadata.get('failure_rate', 0.0)
            }
        elif report_type == 'generation':
            metadata = data.get('metadata', {})
            return {
                'overall_score': metadata.get('overall_score', '0/7'),
                'overall_percentage': metadata.get('overall_percentage', '0%'),
                'pass_rate': metadata.get('pass_rate', '0%'),
                'performance_category': metadata.get('performance_category', 'unknown'),
                'clinical_accuracy_percentage': metadata.get('clinical_accuracy_percentage', '0%'),
                'generation_model': metadata.get('generation_model', 'unknown'),
                'judge_model': metadata.get('judge_model', 'unknown')
            }
        else:
            return {}
    
    def create_comparison_dataframe(self, reports: List[Dict[str, Any]], report_type: str) -> pd.DataFrame:
        """Create a DataFrame for comparing multiple reports.
        
        Args:
            reports: List of report data
            report_type: Type of reports ('retrieval' or 'generation')
            
        Returns:
            Pandas DataFrame with comparison metrics
        """
        if not reports:
            return pd.DataFrame()
        
        comparison_data = []
        
        for report in reports:
            metadata = report.get('metadata', {})
            row = {
                'timestamp': metadata.get('timestamp', 'unknown'),
                'total_queries': metadata.get('total_queries', 0)
            }
            
            if report_type == 'retrieval':
                row.update({
                    'hit_rate_at_1': metadata.get('hit_rate_at_1', 0.0),
                    'hit_rate_at_3': metadata.get('hit_rate_at_3', 0.0),
                    'hit_rate_at_5': metadata.get('hit_rate_at_5', 0.0),
                    'success_rate': metadata.get('success_rate', 0.0),
                    'mean_reciprocal_rank': metadata.get('mean_reciprocal_rank', 0.0),
                    'failure_rate': metadata.get('failure_rate', 0.0),
                    'performance_category': metadata.get('performance_category', 'unknown')
                })
            elif report_type == 'generation':
                # Parse overall percentage from string like "78.3%"
                overall_pct_str = metadata.get('overall_percentage', '0%')
                overall_pct = float(overall_pct_str.replace('%', '')) if '%' in str(overall_pct_str) else float(overall_pct_str or 0)
                
                pass_rate_str = metadata.get('pass_rate', '0%')
                pass_rate = float(pass_rate_str.replace('%', '')) if '%' in str(pass_rate_str) else float(pass_rate_str or 0)
                
                clinical_acc_str = metadata.get('clinical_accuracy_percentage', '0%')
                clinical_acc = float(clinical_acc_str.replace('%', '')) if '%' in str(clinical_acc_str) else float(clinical_acc_str or 0)
                
                row.update({
                    'overall_score': metadata.get('overall_score', '0/7'),
                    'overall_percentage': overall_pct,
                    'pass_rate': pass_rate,
                    'clinical_accuracy_percentage': clinical_acc,
                    'relevance_percentage': metadata.get('relevance_percentage', 0.0),
                    'evidence_support_percentage': metadata.get('evidence_support_percentage', 0.0),
                    'performance_category': metadata.get('performance_category', 'unknown'),
                    'generation_model': metadata.get('generation_model', 'unknown'),
                    'judge_model': metadata.get('judge_model', 'unknown')
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_failure_analysis_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract failure analysis information from a report.
        
        Args:
            report: Full report data
            
        Returns:
            Failure analysis summary
        """
        failure_analysis = report.get('failure_analysis', {})
        detailed_failure = report.get('detailed_failure_analysis', {})
        
        return {
            'total_failures': failure_analysis.get('total_failures', 0),
            'failure_rate': failure_analysis.get('failure_rate', 0.0),
            'failures_by_section': failure_analysis.get('top_failing_sections', detailed_failure.get('failures_by_section', {})),
            'failures_by_document': failure_analysis.get('top_failing_documents', detailed_failure.get('failures_by_document', {})),
            'example_failures': detailed_failure.get('example_failures', detailed_failure.get('low_scoring_examples', []))
        }