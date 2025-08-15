"""
Handles interaction with Ollama LLM models.
"""

import time
import ollama
from functools import lru_cache
from typing import Optional
from app.utils import log_info, log_error, log_warning
from langchain_community.chat_models import ChatOllama
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
        models = ollama.list()
        available_models = [m["name"] for m in models.get("models", [])]

        # Check exact match or partial match (for tags)
        is_available = any(model == m or model in m for m in available_models)

        if not is_available:
            log_warning(
                f"Model '{model}' not found. Available models: {available_models}"
            )
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
    if not check_ollama_connection():
        return "[Error: Cannot connect to Ollama service]"

    # Check model availability
    if not check_model_availability(model):
        return f"[Error: Model '{model}' not available]"

    log_info(f"Generating response with model '{model}' (temp={temperature})...")

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

        # Generate response
        response = ollama.chat(model=model, messages=messages, options=options)

        # Extract and validate response
        if "message" not in response or "content" not in response["message"]:
            log_error("Invalid response format from Ollama")
            return "[Error: Invalid response format]"

        content = response["message"]["content"].strip()

        if not content:
            log_warning("Empty response generated")
            return "[Warning: Empty response generated]"

        return content

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
        models = ollama.list()
        return [m["name"] for m in models.get("models", [])]
    except Exception as e:
        log_error(f"Failed to get available models: {e}")
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
        log_info(
            f"Sending prompt to model '{model}' (temp={temperature}, stop={stop}, seed={seed})"
        )
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        log_error(f"Chat failed: {e}")
        return f"[ERROR] {e}"
