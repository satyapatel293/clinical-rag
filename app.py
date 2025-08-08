#!/usr/bin/env python3
"""
Simple Streamlit frontend for Clinical RAG system.
"""

import os
import streamlit as st

# Set required environment variables for ZenML
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Import clinical RAG components
from pipelines.response_generation_pipeline import clinical_generation_pipeline
from pipelines.simple_generation_pipeline import simple_clinical_generation_pipeline
from utils.search import ClinicalRAGSearcher

# Debug: Check environment variables
openai_key_available = bool(os.getenv('OPENAI_API_KEY'))
if not openai_key_available:
    st.warning("⚠️ OPENAI_API_KEY not found in environment. OpenAI models will not work.")

# Page configuration
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="centered"
)

# Main UI
st.title("🏥 Clinical RAG Assistant")
st.markdown("Ask clinical questions and get evidence-based responses from multiple LLM providers.")

# Navigation info
st.info("💡 **New!** Check out the [Evaluation Results](/evaluation_results) page to view system performance metrics and analysis.")

# Model configuration (no Anthropic as requested)
models = {
    "Ollama Llama 3.2 3B": "ollama/llama3.2:3b",
    "OpenAI GPT-4o": "openai/gpt-4o",
    "OpenAI GPT-3.5 Turbo": "openai/gpt-3.5-turbo"
}

# UI Components
selected_model_name = st.selectbox(
    "Choose a model:",
    list(models.keys()),
    help="Select the language model to use for generating responses"
)

model_name = models[selected_model_name]

# Query input
query = st.text_area(
    "Enter your clinical question:",
    value="",
    height=100,
    placeholder="Example: What exercises help with achilles pain?",
    help="Ask any clinical question. The system will search for relevant evidence and generate a response."
)

# Main query button
if st.button("🔍 Ask Question", type="primary"):
    if query.strip():
        with st.spinner(f"Searching knowledge base and querying {selected_model_name}..."):
            try:
                # Step 1: Search for relevant chunks
                searcher = ClinicalRAGSearcher()
                results = searcher.search_similar_chunks(
                    query=query,
                    top_k=5,
                    similarity_threshold=0.3
                )
                
                # Step 2: Choose pipeline based on context availability
                if results:
                    # Found relevant context - use full RAG pipeline
                    st.info(f"Found {len(results)} relevant chunks. Using RAG pipeline with context.")
                    
                    pipeline_result = clinical_generation_pipeline(
                        query=query,
                        retrieved_chunks=results,
                        model_name=model_name,
                        temperature=0.1
                    )
                else:
                    # No relevant context found - use simple generation
                    st.warning("No relevant clinical information found. Using simple generation without context.")
                    
                    pipeline_result = simple_clinical_generation_pipeline(
                        query=query,
                        model_name=model_name,
                        temperature=0.1
                    )
                
                # Step 3: Extract and display result
                result = pipeline_result.steps["format_final_cli_output"].output.load()
                
                if result['success']:
                    st.success("✅ Response generated successfully!")
                    
                    # Display the response
                    st.markdown(result['cli_output'])
                    
                else:
                    st.error(f"❌ Generation failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ System error: {str(e)}")
                st.info("Please check that:")
                st.markdown("""
                - The database is running and accessible
                - Required API keys are set as environment variables
                - ZenML server is running (if needed)
                """)
    else:
        st.warning("⚠️ Please enter a clinical question before submitting.")

# Sidebar with information
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("""
    - **🏥 [Clinical Q&A](/)**: Ask clinical questions (current page)
    - **📊 [Evaluation Results](/evaluation_results)**: View system performance metrics
    """)
    
    st.markdown("### 🔧 System Information")
    
    st.markdown("**Supported Models:**")
    for name, model in models.items():
        st.markdown(f"- {name}: `{model}`")
    
    st.markdown("### 📚 How it works:")
    st.markdown("""
    1. **Search**: Looks for relevant clinical evidence in the knowledge base
    2. **Context**: If evidence found, includes it as context for the AI
    3. **Generate**: Uses selected model to create evidence-based response
    4. **Fallback**: If no evidence found, generates general clinical response
    """)
    
    st.markdown("### 🔑 Setup Requirements:")
    st.markdown("""
    **For OpenAI models:**
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
    
    **For Ollama models:**
    ```bash
    ollama serve
    ollama pull llama3.2:3b
    ```
    """)
    
    st.markdown("### 💡 Tips:")
    st.markdown("""
    - Ask specific clinical questions for best results
    - Include condition names, symptoms, or treatments
    - Be clear and concise in your queries
    """)
    
    st.markdown("### 📊 Evaluation")
    st.markdown("""
    **Performance Tracking:**
    - Retrieval accuracy (Hit Rate@K)
    - Generation quality (LLM-as-Judge)
    - Clinical accuracy assessment
    - Failure analysis by document/section
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by ZenML, LiteLLM, and clinical evidence from orthopedic literature.*")