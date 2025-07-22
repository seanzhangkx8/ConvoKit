from openai import OpenAI, OpenAIError, RateLimitError, Timeout
from .base import LLMClient, LLMResponse
import time

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt, output_max_tokens=512, temperature=0.0, times_retried=0) -> LLMResponse:
        start = time.time()
        retry_after = 10

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                max_tokens=output_max_tokens,
                temperature=temperature,
            )
        except (OpenAIError, RateLimitError, Timeout) as e:
            if times_retried >= 3:
                raise Exception("Retry failed after multiple attempts.") from e
            print(f"{type(e).__name__}: {e}. Retrying in {retry_after}s...")
            time.sleep(retry_after)
            return self.generate(prompt, output_max_tokens, temperature, times_retried + 1)

        elapsed = time.time() - start
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else -1
        return LLMResponse(text=content, tokens=tokens_used, latency=elapsed, raw=response)
