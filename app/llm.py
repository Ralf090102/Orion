"""
Handles interaction with Ollama LLM models.
"""

import ollama


def generate_response(prompt: str, model: str = "llama3") -> str:
    """
    Sends a prompt to Ollama and returns the generated response.
    """
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
