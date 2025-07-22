from .base import LLMClient, LLMResponse
import time

class LocalClient(LLMClient):
    """
    This is just a template for a local LLM client. Please review the actual implementation
    """
    def __init__(self, model_path: str):
        self.model_path = model_path # e.g., load a llama.cpp or GGUF-backed model

    def generate(self, messages, **kwargs) -> LLMResponse:
        start = time.time()
        prompt = " ".join(m["content"] for m in messages)
        response = f"[Mock local model output for: {prompt}]"
        latency = time.time() - start
        return LLMResponse(text=response, tokens=-1, latency=latency, raw={"prompt": prompt})
