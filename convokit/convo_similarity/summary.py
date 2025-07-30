import os
import ast
import re
from convokit.genai import get_llm_client
from convokit.genai.genai_config import GenAIConfigManager


class SCDWriter:
    """
    A writer class to generate SCD summaries and SoP bulletpoints
    from conversation transcripts using GenAI-backed models.
    """

    SUMMARY_PROMPT_TEMPLATE = None
    BULLETPOINT_PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompts(cls):
        """Lazy load prompts into class variables."""
        if cls.SUMMARY_PROMPT_TEMPLATE is None or cls.BULLETPOINT_PROMPT_TEMPLATE is None:
            base_path = os.path.dirname(__file__)
            with open(os.path.join(base_path, "prompts/scd_prompt.txt"), "r", encoding="utf-8") as f:
                cls.SUMMARY_PROMPT_TEMPLATE = f.read()
            with open(os.path.join(base_path, "prompts/sop_prompt.txt"), "r", encoding="utf-8") as f:
                cls.BULLETPOINT_PROMPT_TEMPLATE = f.read()

    def __init__(self, model_provider: str, model: str = None):
        """
        Initialize the SCD processor with a specified model provider and optional model name. 
        If no model is specified, defaults to our selected default model.
        """
        self.model_provider = model_provider
        self.model = model

        self._load_prompts()

        config = GenAIConfigManager()
        self.client = get_llm_client(model_provider, config, model=model)
    
    def _convert_to_single_quote_dict_string(self, response: str) -> str:
        """
        Clean and normalize GenAI output for safe ast.literal_eval parsing.
        """
        response = response.strip()

        # Normalize possessives and contractions
        response = re.sub(r"'s\b", "s", response)
        response = re.sub(r"'t\b", "t", response)
        response = re.sub(r"'ve\b", "ve", response)

        # Ensure entries are comma-separated
        response = re.sub(r'(")(\s*\'\d+\'\s*:)', r'\1,\2', response)

        # Replace double quotes around values with single quotes
        response = re.sub(
            r'(?<=:\s)"(.*?)"',
            lambda m: "'" + m.group(1).replace("'", "\\'") + "'",
            response,
        )

        return response

    def get_scd_summary(self, transcript: str) -> str:
        """
        Generate the SCD summary from a transcript.
        """
        query = self.SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        response = self.client.generate(query, model=self.model)
        return response.text

    def get_sop_from_summary(self, summary: str) -> dict:
        """
        Generate SoP bulletpoints from an SCD summary.
        """
        query = self.BULLETPOINT_PROMPT_TEMPLATE + summary
        response = self.client.generate(query, model=self.model)

        parsed_text = response.text.strip()

        try:
            return ast.literal_eval(parsed_text)
        except Exception:
            # Fallback: clean and retry
            cleaned_text = self._convert_to_single_quote_dict_string(parsed_text)
            try:
                return ast.literal_eval(cleaned_text)
            except Exception:
                print("output (cleaned): " + cleaned_text)
                raise Exception("The response was not formatted properly for literal_eval parsing")


    def get_scd_and_sop(self, transcript: str) -> tuple[str, dict]:
        """
        Generate both the SCD summary and the SoP bulletpoints in one call.
        """
        summary_text = self.get_scd_summary(transcript)
        bulletpoints = self.get_sop_from_summary(summary_text)
        return summary_text, bulletpoints
