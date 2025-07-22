from .openai_client import OpenAIClient
from .google_client import GeminiClient
from .local_client import LocalClient

def get_llm_client(provider: str, config_manager, **kwargs):
    api_key = config_manager.get_api_key(provider)
    if provider.lower() == "openai":
        return OpenAIClient(api_key=api_key, **kwargs)
    elif provider.lower() == "gemini":
        return GeminiClient(api_key=api_key, **kwargs)
    elif provider.lower() == "local": # should avoid using this before reimplementing local_client.py
        return LocalClient(model_path=kwargs.get("model_path", "./"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")
