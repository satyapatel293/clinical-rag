"""Generate with LiteLLM step for Clinical RAG generation pipeline."""

from typing import Dict, Any
import time
import logging
from zenml import step

logger = logging.getLogger(__name__)


@step
def generate_with_litellm(
    prompt_result: Dict[str, Any],
    model_name: str = "ollama/llama3.2:3b",
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Generate clinical response using LiteLLM (supports multiple providers).
    
    Args:
        prompt_result: Output from build_clinical_prompt step
        model_name: LiteLLM model name (e.g., "openai/gpt-4", "ollama/llama3.2:3b")
        temperature: Generation temperature (0.0-1.0)
        **kwargs: Additional parameters passed to LiteLLM completion
        
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
        
        # Import litellm
        try:
            from litellm import completion
        except ImportError:
            return {
                'generated_response': "",
                'success': False,
                'error': "LiteLLM package not available. Run: pip install litellm"
            }
        
        prompt = prompt_result['prompt']
        
        logger.info(f"Generating response using {model_name}")
        start_time = time.time()
        
        # Generate response using LiteLLM (following your test_model pattern)
        try:
            # Import and configure LiteLLM to drop unsupported parameters
            import litellm
            litellm.drop_params = True  # Automatically drop unsupported parameters
            
            response = completion(
                model=model_name,
                messages=[{"content": prompt, "role": "user"}],
                temperature=temperature,
            )
            
            generation_time = time.time() - start_time
            generated_text = response.choices[0].message.content
            
            # Extract token usage if available
            usage_info = getattr(response, 'usage', None)
            
            generation_metadata = {
                'model_name': model_name,
                'generation_time_seconds': generation_time,
                'temperature': temperature,
                'prompt_length': len(prompt),
                'response_length': len(generated_text),
                'prompt_tokens': getattr(usage_info, 'prompt_tokens', 0) if usage_info else 0,
                'completion_tokens': getattr(usage_info, 'completion_tokens', 0) if usage_info else 0,
                'total_tokens': getattr(usage_info, 'total_tokens', 0) if usage_info else 0,
                'generation_timestamp': time.time()
            }
            
            return {
                'generated_response': generated_text,
                'metadata': {
                    **generation_metadata,
                    'prompt_metadata': prompt_result.get('metadata', {})
                },
                'success': True,
                'error': None
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"LiteLLM generation failed after {generation_time:.2f}s: {e}")
            return {
                'generated_response': "",
                'metadata': {
                    'model_name': model_name,
                    'generation_time_seconds': generation_time,
                    'temperature': temperature,
                    'failed_at': 'litellm_completion'
                },
                'success': False,
                'error': f"LiteLLM generation failed: {str(e)}"
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