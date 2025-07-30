import ast
import numpy as np
import os
import re
from convokit.genai.genai_config import GenAIConfigManager
from convokit.genai import get_llm_client


class NaiveConDynS:
    """
    A class to compute naive ConDynS score between two sequences of events
    """

    NAIVE_CONDYNS_PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompts(cls):
        """Lazy load prompts into class variables."""
        if cls.NAIVE_CONDYNS_PROMPT_TEMPLATE is None:
            base_path = os.path.dirname(__file__)
            with open(os.path.join(base_path, "prompts/naive_condyns_prompt.txt"), "r", encoding="utf-8") as f:
                cls.NAIVE_CONDYNS_PROMPT_TEMPLATE = f.read()

    def __init__(self, model_provider: str, model: str = None):
        """
        Initialize the NaiveConDynS score computer with a specified model provider and optional model name.        
        Args:
            model_provider: The LLM provider to use (e.g., "gpt", "gemini")
            model: Optional specific model name
        """
        self.model_provider = model_provider
        self.model = model

        self._load_prompts()

        config = GenAIConfigManager()
        self.client = get_llm_client(model_provider, config, model=model)

    def _clean_model_output_to_dict(self, text: str) -> dict:
        """
        Clean and parse model output into a dictionary.
        """
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid dictionary boundaries found.")
        
        dict_str = text[start:end+1]
        dict_str = re.sub(r"'s\b", "s", dict_str)
        dict_str = re.sub(r"'t\b", "t", dict_str)
        dict_str = re.sub(r"'ve\b", "ve", dict_str)
        return ast.literal_eval(dict_str)

    def compute_unidirectional_naive_condyns(self, sop1, sop2):
        """
        Compute unidirectional naive conditional dynamics similarity between two sequences of events.
        
        Args:
            sop1: Dictionary of events from the first conversation's sop with ordered keys ('0', '1', etc.)
            sop2: Dictionary of events from the second conversation's sop with ordered keys ('0', '1', etc.)
        
        Returns:
            Dictionary with analysis and scores for each event in sop1
        """
        # Format the prompt with the two sequences of events
        full_prompt = self.NAIVE_CONDYNS_PROMPT_TEMPLATE.format(sop1=sop1, sop2=sop2)
        
        response = self.client.generate(full_prompt)
        try:
            response_dict = self._clean_model_output_to_dict(response.text)
        except (SyntaxError, ValueError) as e:
            print(response.text)
            print("Error parsing output:", e)
            raise Exception("error parsing")
        return response_dict
        
    def compute_bidirectional_naive_condyns(self, sop1, sop2):
        """
        Compute bidirectional naive conditional dynamics similarity between two sequences of events.
        
        Args:
            sop1: Dictionary of events from the first sequence with ordered keys ('0', '1', etc.)
            sop2: Dictionary of events from the second sequence with ordered keys ('0', '1', etc.)
        
        Returns:
            List of [response_dict1, response_dict2] where each dict contains
            analysis and scores for each event
        """
        response_dict1 = self.compute_unidirectional_naive_condyns(sop1, sop2)
        response_dict2 = self.compute_unidirectional_naive_condyns(sop2, sop1)
        return [response_dict1, response_dict2]

    def measure_score(self, data):
        """
        Calculate the mean score from a similarity result dictionary.
        """
        sum_score = []
        for item in data.values():
            sum_score.append(item['score'])
        return np.mean(sum_score)
        
    def compute_score_from_results(self, results):
        """
        Compute scores from bidirectional similarity results.
        """
        scores = []
        for result in results:
            scores.append(self.measure_score(result))
        return scores