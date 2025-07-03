"""Clinical response generation using Ollama LLM."""

import logging
from typing import List, Dict, Any, Optional
import ollama
from .config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class ClinicalResponseGenerator:
    """Generates clinical responses using local Ollama LLM."""
    
    def __init__(self, model_name: str = "llama3.2:3b", temperature: float = 0.1):
        """
        Initialize the clinical response generator.
        
        Args:
            model_name: Ollama model name (default: llama3.2:3b)
            temperature: Generation temperature for consistency (default: 0.1)
        """
        self.model_name = model_name
        self.temperature = temperature
        self._verify_model_availability()
    
    def _verify_model_availability(self) -> None:
        """Verify that the specified model is available in Ollama."""
        try:
            models_response = ollama.list()
            available_models = [model.model for model in models_response.models]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                raise ValueError(f"Model {self.model_name} not available. Run: ollama pull {self.model_name}")
            
            logger.info(f"✅ Model {self.model_name} is available")
            
        except Exception as e:
            logger.error(f"Failed to verify model availability: {e}")
            raise
    
    def _build_clinical_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a clinical prompt with retrieved context.
        
        Args:
            query: User's clinical query
            chunks: Retrieved text chunks with metadata
            
        Returns:
            Formatted prompt for LLM
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            similarity = chunk.get('similarity', 0.0)
            section = chunk.get('section_type', 'unknown')
            content = chunk.get('content', '')
            
            context_parts.append(f"[Source {i} - {section.title()} - Relevance: {similarity:.2f}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a clinical decision support assistant for orthopedic physical therapy. Your role is to provide evidence-based guidance to healthcare professionals.

CLINICAL CONTEXT:
{context}

CLINICAL QUERY: {query}

INSTRUCTIONS:
1. Provide a clear, evidence-based response based ONLY on the provided clinical context
2. If the context doesn't fully address the query, clearly state what information is missing
3. Include relevant clinical considerations and contraindications when applicable
4. Use professional medical terminology appropriate for healthcare providers
5. Structure your response with clear sections (Assessment, Recommendations, Considerations)
6. If multiple sources conflict, acknowledge the discrepancy
7. Do not provide information not supported by the given context

IMPORTANT: This is decision support for healthcare professionals, not direct patient care advice.

RESPONSE:"""
        
        return prompt
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate clinical response based on query and retrieved chunks.
        
        Args:
            query: User's clinical question
            retrieved_chunks: List of retrieved chunks with metadata
            
        Returns:
            Dictionary with generated response and metadata
        """
        try:
            if not retrieved_chunks:
                return {
                    'success': False,
                    'error': 'No relevant clinical information found',
                    'response': 'I could not find relevant clinical information to answer your query. Please try rephrasing your question or consult additional clinical resources.',
                    'metadata': {
                        'chunks_used': 0,
                        'model': self.model_name,
                        'temperature': self.temperature
                    }
                }
            
            # Build clinical prompt
            prompt = self._build_clinical_prompt(query, retrieved_chunks)
            
            # Generate response using Ollama
            logger.info(f"Generating response using {self.model_name}")
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': self.temperature,
                    'top_p': 0.9,
                    'top_k': 40
                }
            )
            
            generated_text = response['message']['content']
            
            # Extract metadata from response
            usage_info = response.get('usage', {})
            
            return {
                'success': True,
                'response': generated_text,
                'metadata': {
                    'chunks_used': len(retrieved_chunks),
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'prompt_tokens': usage_info.get('prompt_tokens', 0),
                    'completion_tokens': usage_info.get('completion_tokens', 0),
                    'total_tokens': usage_info.get('total_tokens', 0),
                    'chunk_sources': [
                        {
                            'similarity': chunk.get('similarity', 0.0),
                            'section': chunk.get('section_type', 'unknown'),
                            'document': chunk.get('document_name', 'unknown')
                        }
                        for chunk in retrieved_chunks
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'I encountered an error while generating the response. Please try again or consult your clinical resources.',
                'metadata': {
                    'chunks_used': len(retrieved_chunks) if retrieved_chunks else 0,
                    'model': self.model_name,
                    'temperature': self.temperature
                }
            }
    
    def generate_simple_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response without retrieved context (for testing).
        
        Args:
            query: Clinical question
            
        Returns:
            Dictionary with generated response
        """
        try:
            prompt = f"""You are a clinical decision support assistant for orthopedic physical therapy. 

QUERY: {query}

Provide a brief, evidence-based response. Note that this response is generated without specific clinical context and should be verified with current clinical guidelines.

RESPONSE:"""
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': self.temperature
                }
            )
            
            return {
                'success': True,
                'response': response['message']['content'],
                'metadata': {
                    'model': self.model_name,
                    'context_used': False
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate simple response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': 'Failed to generate response.',
                'metadata': {
                    'model': self.model_name,
                    'context_used': False
                }
            }