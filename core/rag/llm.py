"""
Handles interaction with Ollama LLM models.
"""

import time
import ollama
from functools import lru_cache
from typing import Optional
from core.utils.orion_utils import log_info, log_error, log_warning, log_debug
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

# Cache connection check for 5 seconds
_last_check_time = 0
_last_check_result = False

DEFAULT_LLM_MODEL = "mistral"


@lru_cache(maxsize=128)
def model_exists(model: str) -> bool:
    """
    Check if the specified model exists in Ollama.

    Args:
        model: The name of the model to check.

    Returns:
        True if the model exists, False otherwise
    """
    try:
        # show fails fast if model is missing
        ollama.show(model)
        return True
    except Exception:
        return False


def check_ollama_connection() -> bool:
    """
    Check if Ollama service is running and accessible.

    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        ollama.list()
        return True
    except Exception as e:
        log_error(f"Cannot connect to Ollama: {e}")
        log_info("Make sure Ollama is running (ollama serve)")
        return False


def check_ollama_connection_cached(model_name: str) -> bool:
    """
    Check if Ollama service is running and accessible (cached version).

    Args:
        model_name: The name of the model to check.

    Returns:
        True if Ollama is accessible, False otherwise
    """
    global _last_check_time, _last_check_result
    now = time.time()
    if now - _last_check_time < 5:
        return _last_check_result
    try:
        # Simple ping — will raise if Ollama isn't running
        _ = ChatOllama(model=model_name)
        _last_check_result = True
    except Exception as e:
        log_error(f"Ollama connection failed: {e}")
        _last_check_result = False
    _last_check_time = now
    return _last_check_result


def check_model_availability(model: str) -> bool:
    """
    Check if the specified model is available in Ollama.

    Args:
        model: Model name to check

    Returns:
        True if model is available, False otherwise
    """
    if model_exists(model):
        return True

    try:
        # Use the fixed get_available_models function
        available_models = get_available_models()

        # Check exact match or partial match (for tags)
        is_available = any(model == m or model in m for m in available_models)

        if not is_available:
            log_warning(f"Model '{model}' not found. Available models: {available_models}")
            log_info(f"To install the model, run: ollama pull {model}")

        return is_available

    except Exception as e:
        log_error(f"Failed to check model availability: {e}")
        return False


def generate_response(
    prompt: str,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_message: Optional[str] = None,
    stream: bool = False,
    on_token=None,
    num_ctx: Optional[int] = None,
) -> str:
    """
    Sends a prompt to Ollama and returns the generated response.
    Includes comprehensive error handling, model validation, and customizable parameters.

    Args:
        prompt: The input prompt for the model
        model: Ollama model name to use
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate (None for model default)
        system_message: Optional system message to set context

    Returns:
        Generated response text or error message
    """
    if not prompt.strip():
        log_error("Empty prompt provided")
        return "[Error: Empty prompt]"

    # Check Ollama connection
    if not check_ollama_connection_cached(model):
        return "[Error: Cannot connect to Ollama service]"

    # Check model availability
    if not check_model_availability(model):
        return f"[Error: Model '{model}' not available]"

    log_debug(f"Generating response with model '{model}' (temp={temperature})...")

    try:
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Prepare options
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        # Generate response
        if stream:
            content = ""
            for chunk in ollama.chat(model=model, messages=messages, options=options, stream=True):
                part = chunk.get("message", {}).get("content", "")
                if part:
                    if on_token:
                        on_token(part)
                    content += part
            return content.strip() or "[Warning: Empty response generated]"
        else:
            response = ollama.chat(model=model, messages=messages, options=options)
            content = response.get("message", {}).get("content", "").strip()
            return content or "[Warning: Empty response generated]"

    except ollama.ResponseError as e:
        log_error(f"Ollama API error: {e}")
        return f"[Error: Ollama API error - {e}]"

    except Exception as e:
        log_error(f"LLM generation failed: {e}")
        return "[Error: Failed to generate response]"


def get_available_models() -> list:
    """
    Get list of available Ollama models.

    Returns:
        List of available model names
    """
    try:
        models_response = ollama.list()
        log_debug(f"Raw ollama.list() response: {models_response}")
        log_debug(f"Response type: {type(models_response)}")

        # Handle ListResponse object (newer Ollama API)
        if hasattr(models_response, "models"):
            models = models_response.models
            log_debug(f"Found {len(models)} models in ListResponse object")

            model_names = []
            for model in models:
                if hasattr(model, "model"):
                    # Model object with 'model' attribute
                    model_names.append(model.model)
                    log_debug(f"Added model: {model.model}")
                elif hasattr(model, "name"):
                    # Model object with 'name' attribute (fallback)
                    model_names.append(model.name)
                    log_debug(f"Added model with name: {model.name}")
                elif isinstance(model, dict):
                    # Dictionary fallback
                    name = model.get("name") or model.get("model") or model.get("id")
                    if name:
                        model_names.append(name)
                        log_debug(f"Added model from dict: {name}")
                else:
                    log_warning(f"Unknown model format: {model} (type: {type(model)})")

            return model_names

        # Handle dictionary response (older Ollama API)
        elif isinstance(models_response, dict):
            if "models" in models_response:
                models = models_response["models"]
            else:
                # Sometimes the response is just the models list directly
                models = models_response
        else:
            # If it's already a list
            models = models_response

        # Extract model names from dictionary/list format
        model_names = []
        for model in models:
            if isinstance(model, dict):
                # Try different possible key names
                name = model.get("name") or model.get("model") or model.get("id")
                if name:
                    model_names.append(name)
            elif isinstance(model, str):
                model_names.append(model)

        log_info(f"Extracted model names: {model_names}")
        return model_names

    except Exception as e:
        log_error(f"Failed to get available models: {e}")
        import traceback

        log_error(f"Full traceback: {traceback.format_exc()}")
        return []


def chat(
    prompt: str,
    model: str = "mistral",
    temperature: float = 0.7,
    stop: list[str] = None,
    seed: int = None,
) -> str:
    """
    Send a prompt to Ollama and get a response.

    Args:
        prompt: The input text for the model.
        model: The Ollama model name (default: mistral).
        temperature: Sampling temperature (lower = more deterministic).
        stop: Optional list of stop sequences.
        seed: Optional random seed for reproducibility in tests.

    Returns:
        Model's response text.
    """
    if not check_ollama_connection_cached(model):
        return "[ERROR] Ollama is not running or model is unavailable."

    try:
        chat_model = ChatOllama(
            model=model,
            temperature=temperature,
            stop=stop or [],
            # Ollama supports seed as a generation option
            seed=seed,
        )
        log_info(f"Sending prompt to model '{model}' (temp={temperature}, stop={stop}, seed={seed})")
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        log_error(f"Chat failed: {e}")
        return f"[ERROR] {e}"


def chat_stream(
    prompt: str,
    model: str = "mistral",
    temperature: float = 0.7,
    stop: list[str] = None,
    seed: int = None,
):
    """
    Stream a response from Ollama (generator for GUI/real-time display).

    Args:
        prompt: The input text for the model.
        model: The Ollama model name (default: mistral).
        temperature: Sampling temperature (lower = more deterministic).
        stop: Optional list of stop sequences.
        seed: Optional random seed for reproducibility in tests.

    Yields:
        String chunks of the model's response.
    """
    if not check_ollama_connection_cached(model):
        yield "[ERROR] Ollama is not running or model is unavailable."
        return

    try:
        chat_model = ChatOllama(
            model=model,
            temperature=temperature,
            stop=stop or [],
            seed=seed,
        )
        log_info(f"Streaming from model '{model}'")

        for chunk in chat_model.stream([HumanMessage(content=prompt)]):
            yield chunk.content

    except Exception as e:
        log_error(f"Chat streaming failed: {e}")
        yield f"[ERROR] {e}"
