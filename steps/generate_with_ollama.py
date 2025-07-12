"""Generate with Ollama LLM step for Clinical RAG generation pipeline."""

from typing import Dict, Any
import time
import logging
from zenml import step

logger = logging.getLogger(__name__)


#@step
def generate_with_ollama(
    prompt_result: Dict[str, Any],
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 40
) -> Dict[str, Any]:
    """
    Generate clinical response using Ollama LLM.
    
    Args:
        prompt_result: Output from build_clinical_prompt step
        model_name: Ollama model name to use
        temperature: Generation temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        Dictionary containing generated response and metadata
    """
    try:
        if not prompt_result['success']:
            return {
                'generated_response': "",
                'success': False,
                'error': f"Cannot generate - prompt building failed: {prompt_result['error']}"
            }
        
        # Import ollama here to avoid unnecessary loading
        try:
            import ollama
        except ImportError:
            return {
                'generated_response': "",
                'success': False,
                'error': "Ollama package not available. Run: pip install ollama"
            }
        
        prompt = prompt_result['prompt']
        
        # Verify model availability
        try:
            models_response = ollama.list()
            available_models = [model.model for model in models_response.models]
            
            if model_name not in available_models:
                return {
                    'generated_response': "",
                    'success': False,
                    'error': f"Model {model_name} not available. Available: {available_models}"
                }
        except Exception as e:
            return {
                'generated_response': "",
                'success': False,
                'error': f"Failed to verify model availability: {str(e)}"
            }
        
        logger.info(f"Generating response using {model_name}")
        start_time = time.time()
        
        # Generate response using Ollama
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k
                }
            )
            
            generation_time = time.time() - start_time
            generated_text = response['message']['content']
            
            # Extract generation metadata
            usage_info = response.get('usage', {})
            
            generation_metadata = {
                'model_name': model_name,
                'generation_time_seconds': generation_time,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'prompt_length': len(prompt),
                'response_length': len(generated_text),
                'prompt_tokens': usage_info.get('prompt_tokens', 0),
                'completion_tokens': usage_info.get('completion_tokens', 0),
                'total_tokens': usage_info.get('total_tokens', 0),
                'generation_timestamp': time.time()
            }
            
            return {
                'generated_response': generated_text,
                'metadata': {
                    **generation_metadata,
                    'prompt_metadata': prompt_result.get('metadata', {}),
                    'ollama_response': {
                        'model': response.get('model', model_name),
                        'created_at': response.get('created_at'),
                        'done': response.get('done', True)
                    }
                },
                'success': True,
                'error': None
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Ollama generation failed after {generation_time:.2f}s: {e}")
            return {
                'generated_response': "",
                'metadata': {
                    'model_name': model_name,
                    'generation_time_seconds': generation_time,
                    'temperature': temperature,
                    'failed_at': 'ollama_chat'
                },
                'success': False,
                'error': f"Ollama generation failed: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Generation step failed: {e}")
        return {
            'generated_response': "",
            'metadata': {
                'model_name': model_name,
                'temperature': temperature,
                'failed_at': 'step_initialization'
            },
            'success': False,
            'error': f"Generation step failed: {str(e)}"
        }


#@step
def verify_ollama_model(model_name: str = "llama3.2:3b") -> Dict[str, Any]:
    """
    Verify that Ollama model is available before generation.
    
    Args:
        model_name: Model name to verify
        
    Returns:
        Dictionary with verification results
    """
    try:
        import ollama
        
        models_response = ollama.list()
        available_models = [model.model for model in models_response.models]
        
        if model_name in available_models:
            # Get model details
            model_details = None
            for model in models_response.models:
                if model.model == model_name:
                    model_details = {
                        'model': model.model,
                        'digest': model.digest,
                        'size': model.size,
                        'modified_at': str(model.modified_at)
                    }
                    break
            
            return {
                'model_available': True,
                'model_name': model_name,
                'model_details': model_details,
                'available_models': available_models,
                'success': True,
                'error': None
            }
        else:
            return {
                'model_available': False,
                'model_name': model_name,
                'available_models': available_models,
                'success': False,
                'error': f"Model {model_name} not found. Run: ollama pull {model_name}"
            }
            
    except Exception as e:
        return {
            'model_available': False,
            'model_name': model_name,
            'success': False,
            'error': f"Model verification failed: {str(e)}"
        }