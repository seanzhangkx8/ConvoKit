from .gpt_client import GPTClient
from .gemini_client import GeminiClient
from .local_client import LocalClient

def get_llm_client(provider: str, config_manager, **kwargs):
    """Factory function as a unified interface to create LLM client instances.
    
    Creates and returns the appropriate LLM client based on the provider name.
    The client is initialized with API keys from the config manager and any additional parameters.
    
    :param provider: Name of the LLM provider ("gpt", "gemini", "local")
    :param config_manager: Configuration manager instance to retrieve API keys
    :param **kwargs: Additional parameters to pass to the client constructor
    :return: Initialized LLM client instance
    :raises ValueError: If the provider is not supported
    """
    api_key = config_manager.get_api_key(provider)
    if provider.lower() == "gpt":
        return GPTClient(api_key=api_key, **kwargs)
    elif provider.lower() == "gemini":
        return GeminiClient(api_key=api_key, **kwargs)
    elif provider.lower() == "local": # should avoid using this before reimplementing local_client.py
        return LocalClient(model_path=kwargs.get("model_path", "./"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")
