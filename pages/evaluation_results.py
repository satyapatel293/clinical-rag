#!/usr/bin/env python3
"""
Streamlit page for displaying evaluation results.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set required environment variables for ZenML
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES' 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import evaluation utilities
from utils.evaluation_loader import EvaluationLoader

# Page configuration
st.set_page_config(
    page_title="Evaluation Results - Clinical RAG",
    page_icon="📊",
    layout="wide"
)

# Initialize evaluation loader
@st.cache_data
def load_evaluation_data():
    """Load evaluation data with caching."""
    loader = EvaluationLoader()
    return loader.get_available_reports(), loader

def format_percentage(value, is_percentage_string=False):
    """Format percentage values consistently."""
    if is_percentage_string and isinstance(value, str):
        return value
    elif isinstance(value, (int, float)):
        return f"{value:.1f}%"
    else:
        return "0.0%"

def format_score(score_str):
    """Format score strings like '5.5/7' consistently."""
    if isinstance(score_str, str) and '/' in score_str:
        return score_str
    else:
        return str(score_str)

# Main page content
st.title("📊 Evaluation Results Dashboard")
st.markdown("Explore comprehensive evaluation results for the Clinical RAG system.")

# Load evaluation data
try:
    reports, loader = load_evaluation_data()
    
    if not reports['retrieval'] and not reports['generation']:
        st.warning("📭 No evaluation reports found. Please run evaluations first.")
        st.markdown("""
        **To generate evaluation reports, run:**
        ```bash
        # For retrieval evaluation
        python evaluation/run_evaluation.py --output evaluation/reports/evaluation_detailed_$(date +%Y%m%d_%H%M%S).json
        
        # For generation evaluation (requires ZenML pipeline)
        python -c "from pipelines.generation_evaluation_pipeline import generation_evaluation_pipeline; generation_evaluation_pipeline()"
        ```
        """)
        st.stop()
    
    # Create tabs for different evaluation types
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Retrieval Results", "🤖 Generation Results"])
    
    with tab1:
        st.header("📈 Evaluation Overview")
        
        # Summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Retrieval Reports",
                len(reports['retrieval']),
                help="Number of retrieval evaluation reports available"
            )
        
        with col2:
            st.metric(
                "Total Generation Reports", 
                len(reports['generation']),
                help="Number of generation evaluation reports available"
            )
        
        with col3:
            if reports['retrieval']:
                latest_retrieval = loader.get_latest_report('retrieval')
                if latest_retrieval:
                    hit_rate = latest_retrieval.get('metadata', {}).get('hit_rate_at_5', 0.0)
                    st.metric(
                        "Latest Hit Rate@5",
                        f"{hit_rate:.1%}",
                        help="Most recent retrieval success rate at top-5"
                    )
                else:
                    st.metric("Latest Hit Rate@5", "N/A")
            else:
                st.metric("Latest Hit Rate@5", "N/A")
        
        with col4:
            if reports['generation']:
                latest_generation = loader.get_latest_report('generation')
                if latest_generation:
                    overall_pct = latest_generation.get('metadata', {}).get('overall_percentage', '0%')
                    st.metric(
                        "Latest Generation Score",
                        overall_pct,
                        help="Most recent overall generation quality percentage"
                    )
                else:
                    st.metric("Latest Generation Score", "N/A")
            else:
                st.metric("Latest Generation Score", "N/A")
        
        # Recent reports summary
        st.subheader("🕒 Recent Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if reports['retrieval']:
                st.markdown("**📊 Latest Retrieval Reports**")
                for report_meta in reports['retrieval'][:3]:  # Show top 3
                    perf_summary = report_meta['performance_summary']
                    st.markdown(f"""
                    - **{report_meta['timestamp']}**: {perf_summary['performance_category']} 
                      (Hit@5: {perf_summary['hit_rate_at_5']:.1%})
                    """)
            else:
                st.markdown("**📊 No Retrieval Reports Found**")
        
        with col2:
            if reports['generation']:
                st.markdown("**🤖 Latest Generation Reports**")
                for report_meta in reports['generation'][:3]:  # Show top 3
                    perf_summary = report_meta['performance_summary']
                    st.markdown(f"""
                    - **{report_meta['timestamp']}**: {perf_summary['performance_category']} 
                      ({perf_summary['overall_percentage']})
                    """)
            else:
                st.markdown("**🤖 No Generation Reports Found**")
    
    with tab2:
        st.header("🔍 Retrieval Evaluation Results")
        
        if not reports['retrieval']:
            st.info("No retrieval evaluation reports found.")
            st.stop()
        
        # Report selector
        retrieval_options = [f"{r['timestamp']} ({r['performance_summary']['performance_category']})" 
                           for r in reports['retrieval']]
        selected_retrieval = st.selectbox(
            "Select Retrieval Report:",
            options=range(len(retrieval_options)),
            format_func=lambda x: retrieval_options[x],
            help="Choose a retrieval evaluation report to analyze"
        )
        
        # Load selected report
        selected_retrieval_report = loader.load_report(reports['retrieval'][selected_retrieval]['file_path'])
        
        if selected_retrieval_report:
            metadata = selected_retrieval_report['metadata']
            
            # Key metrics
            st.subheader("📊 Key Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Hit Rate @ 1", f"{metadata.get('hit_rate_at_1', 0):.1%}")
            with metric_col2:
                st.metric("Hit Rate @ 3", f"{metadata.get('hit_rate_at_3', 0):.1%}")
            with metric_col3:
                st.metric("Hit Rate @ 5", f"{metadata.get('hit_rate_at_5', 0):.1%}")
            with metric_col4:
                st.metric("Mean Reciprocal Rank", f"{metadata.get('mean_reciprocal_rank', 0):.3f}")
            
            # Performance visualization
            st.subheader("📈 Performance Visualization")
            
            # Hit rates chart
            hit_rates_data = {
                'k': [1, 3, 5, 10],
                'Hit Rate': [
                    metadata.get('hit_rate_at_1', 0),
                    metadata.get('hit_rate_at_3', 0), 
                    metadata.get('hit_rate_at_5', 0),
                    metadata.get('hit_rate_at_10', 0)
                ]
            }
            
            fig_hit_rates = px.bar(
                hit_rates_data, 
                x='k', 
                y='Hit Rate',
                title="Hit Rate @ K",
                labels={'Hit Rate': 'Hit Rate (%)', 'k': 'Top K Results'}
            )
            fig_hit_rates.update_traces(marker_color='lightblue')
            fig_hit_rates.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_hit_rates, use_container_width=True)
            
            # Failure analysis
            st.subheader("🔍 Failure Analysis")
            failure_analysis = loader.get_failure_analysis_summary(selected_retrieval_report)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Failures by Section Type**")
                if failure_analysis['failures_by_section']:
                    section_df = pd.DataFrame([
                        {'Section': k, 'Failures': v} 
                        for k, v in failure_analysis['failures_by_section'].items()
                    ])
                    fig_sections = px.pie(
                        section_df, 
                        values='Failures', 
                        names='Section',
                        title="Failure Distribution by Section"
                    )
                    st.plotly_chart(fig_sections, use_container_width=True)
                else:
                    st.info("No section-based failure data available")
            
            with col2:
                st.markdown("**Failures by Document**")
                if failure_analysis['failures_by_document']:
                    doc_df = pd.DataFrame([
                        {'Document': k.replace('.pdf', ''), 'Failures': v}
                        for k, v in failure_analysis['failures_by_document'].items()
                    ])
                    fig_docs = px.bar(
                        doc_df, 
                        x='Failures', 
                        y='Document',
                        orientation='h',
                        title="Failure Distribution by Document"
                    )
                    st.plotly_chart(fig_docs, use_container_width=True)
                else:
                    st.info("No document-based failure data available")
            
            # Example failures
            if failure_analysis['example_failures']:
                st.subheader("❌ Example Failures")
                st.markdown("Queries that failed to retrieve the expected content:")
                
                for i, failure in enumerate(failure_analysis['example_failures'][:5]):  # Show top 5
                    with st.expander(f"Failure {i+1}: {failure['question'][:80]}..."):
                        st.markdown(f"**Question:** {failure['question']}")
                        st.markdown(f"**Expected Chunk ID:** {failure['expected_chunk_id']}")
                        st.markdown(f"**Section Type:** {failure['section_type']}")
                        st.markdown(f"**Document:** {failure['filename']}")
                        if failure.get('retrieved_chunk_ids'):
                            st.markdown(f"**Retrieved Chunk IDs:** {failure['retrieved_chunk_ids']}")
                        if failure.get('top_similarity'):
                            st.markdown(f"**Top Similarity:** {failure['top_similarity']:.3f}")
    
    with tab3:
        st.header("🤖 Generation Evaluation Results")
        
        if not reports['generation']:
            st.info("No generation evaluation reports found.")
            st.stop()
        
        # Report selector
        generation_options = [f"{r['timestamp']} ({r['performance_summary']['performance_category']})" 
                            for r in reports['generation']]
        selected_generation = st.selectbox(
            "Select Generation Report:",
            options=range(len(generation_options)),
            format_func=lambda x: generation_options[x],
            help="Choose a generation evaluation report to analyze"
        )
        
        # Load selected report
        selected_generation_report = loader.load_report(reports['generation'][selected_generation]['file_path'])
        
        if selected_generation_report:
            metadata = selected_generation_report['metadata']
            
            # Key metrics
            st.subheader("📊 Key Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Overall Score", format_score(metadata.get('overall_score', '0/7')))
            with metric_col2:
                st.metric("Overall Percentage", format_percentage(metadata.get('overall_percentage', '0%'), True))
            with metric_col3:
                st.metric("Pass Rate", format_percentage(metadata.get('pass_rate', '0%'), True))
            with metric_col4:
                st.metric("Clinical Accuracy", format_percentage(metadata.get('clinical_accuracy_percentage', '0%'), True))
            
            # Model information
            st.subheader("🔧 Model Configuration")
            model_col1, model_col2 = st.columns(2)
            with model_col1:
                st.info(f"**Generation Model:** {metadata.get('generation_model', 'Unknown')}")
            with model_col2:
                st.info(f"**Judge Model:** {metadata.get('judge_model', 'Unknown')}")
            
            # Dimension scores visualization
            st.subheader("📈 Performance by Dimension")
            
            dimensions_data = {
                'Dimension': ['Clinical Accuracy', 'Relevance', 'Evidence Support'],
                'Score (%)': [
                    float(str(metadata.get('clinical_accuracy_percentage', '0')).replace('%', '')),
                    float(str(metadata.get('relevance_percentage', '0')).replace('%', '')),
                    float(str(metadata.get('evidence_support_percentage', '0')).replace('%', ''))
                ]
            }
            
            fig_dimensions = px.bar(
                dimensions_data,
                x='Dimension',
                y='Score (%)',
                title="Performance by Evaluation Dimension",
                color='Score (%)',
                color_continuous_scale='RdYlGn'
            )
            fig_dimensions.update_layout(showlegend=False)
            st.plotly_chart(fig_dimensions, use_container_width=True)
            
            # Failure analysis
            st.subheader("🔍 Failure Analysis")
            failure_analysis = loader.get_failure_analysis_summary(selected_generation_report)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Failures by Section Type**")
                if failure_analysis['failures_by_section']:
                    section_df = pd.DataFrame([
                        {'Section': k, 'Failures': v}
                        for k, v in failure_analysis['failures_by_section'].items()
                    ])
                    fig_sections = px.pie(
                        section_df,
                        values='Failures',
                        names='Section', 
                        title="Failure Distribution by Section"
                    )
                    st.plotly_chart(fig_sections, use_container_width=True)
                else:
                    st.info("No section-based failure data available")
            
            with col2:
                st.markdown("**Failures by Document**")
                if failure_analysis['failures_by_document']:
                    doc_df = pd.DataFrame([
                        {'Document': k.replace('.pdf', ''), 'Failures': v}
                        for k, v in failure_analysis['failures_by_document'].items()
                    ])
                    fig_docs = px.bar(
                        doc_df,
                        x='Failures',
                        y='Document',
                        orientation='h',
                        title="Failure Distribution by Document"
                    )
                    st.plotly_chart(fig_docs, use_container_width=True)
                else:
                    st.info("No document-based failure data available")
            
            # Low scoring examples
            if failure_analysis['example_failures']:
                st.subheader("📉 Low Scoring Examples")
                st.markdown("Generated responses that received low evaluation scores:")
                
                for i, example in enumerate(failure_analysis['example_failures'][:3]):  # Show top 3
                    with st.expander(f"Example {i+1}: Score {example.get('judge_score', 0)}/7"):
                        st.markdown(f"**Question:** {example['question']}")
                        st.markdown(f"**Section Type:** {example['section_type']}")
                        st.markdown(f"**Document:** {example['filename']}")
                        st.markdown(f"**Judge Score:** {example.get('judge_score', 0)}/7 ({example.get('judge_percentage', 0):.1f}%)")
                        
                        if example.get('dimension_scores'):
                            scores = example['dimension_scores']
                            st.markdown(f"**Dimension Scores:**")
                            st.markdown(f"- Clinical Accuracy: {scores.get('clinical_accuracy', 0)}/3")
                            st.markdown(f"- Relevance: {scores.get('relevance', 0)}/2")
                            st.markdown(f"- Evidence Support: {scores.get('evidence_support', 0)}/2")
                        
                        if st.button(f"Show Generated Response {i+1}", key=f"show_response_{i}"):
                            st.markdown("**Generated Response:**")
                            st.text_area(
                                "Response Content:",
                                example.get('generated_response', 'No response available'),
                                height=200,
                                key=f"response_text_{i}"
                            )

except Exception as e:
    st.error(f"❌ Error loading evaluation data: {str(e)}")
    st.markdown("""
    **Troubleshooting:**
    - Ensure evaluation reports exist in `evaluation/reports/` directory
    - Check that report files are valid JSON format
    - Verify file permissions are correct
    """)

# Footer
st.markdown("---")
st.markdown("*Clinical RAG Evaluation Dashboard - Powered by Streamlit*")