from abc import ABC, abstractmethod
import time

class LLMResponse:
    def __init__(self, text: str, tokens: int, latency: float, raw: dict):
        self.text = text
        self.tokens = tokens
        self.latency = latency
        self.raw = raw

    def __repr__(self):
        return f"LLMResponse(text={self.text[:30]}..., tokens={self.tokens}, latency={self.latency:.2f}s)"

class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages, **kwargs) -> LLMResponse:
        pass

    def stream(self, messages, callback, **kwargs):
        raise NotImplementedError("Streaming not supported yet.")
