from evaluate import load
from sentence_transformers import SentenceTransformer, util
from convokit.genai.genai_config import GenAIConfigManager
from convokit.genai import get_llm_client
import re
import json

config = GenAIConfigManager()
client = get_llm_client("gpt", config)

st_model = SentenceTransformer("all-MiniLM-L6-v2", device = "cuda")
bertscore = load("bertscore")


def get_bertscore(pred, ref):
    a = [pred]
    b = [ref]
    return bertscore.compute(predictions=a, references=b, model_type="distilbert-base-uncased")


def get_cosine_similarity(pred, ref):
    embeddings = st_model.encode([pred, ref], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()


def parse_gpt_responses(response):
    clean_json_str = re.sub(r"```json|```", "", response).strip()
    try:
        parsed_data = json.loads(clean_json_str)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for response: {response}")  # Debugging output if needed
    return parsed_data


def get_gpt_compare_score(pred, ref, prompt, model_name="chatgpt-4o-latest"):
    gpt_prompt = prompt.format(pred=pred, ref=ref)
    response = client.generate(gpt_prompt)
    parsed_response = parse_gpt_responses(response)
    score = parsed_response['sim_score']
    reason = parsed_response['reason']
    return score, reason


def get_naive_gpt_compare_score_SCDs(x, y):
    naive_gpt_compare_scd_prompt = """Compare the following two summary of conversation dynamics (SCD) of two online conversations, rate the similarity of the two conversations on a scale from 1 to 100, based on their persuasion trajectory reflected in the SCDs.
    
### **Key Aspects of Persuasion Trajectory**
- **Persuasion Strategies**: Logical reasoning, emotional appeals, rhetorical questions, citing authority, anecdotes, hypothetical scenarios, refuting counterarguments, shifting burden of proof, repetition, framing, social proof.
- **Interaction Dynamics**: Engagement patterns (e.g., single argument vs. back-and-forth), timing effects, persistence vs. resistance.
- **Response to Persuasion**: Agreement, concession, skepticism, counter-argument, disengagement, linguistic indicators (e.g., hedging, intensity, pronouns).
- **Trajectory of Persuasion**: Gradual shift, immediate agreement, persistent resistance, partial concession, reversal, stalemate.

### **Ignore**:
- **Do not consider specific topics, claims, or arguments.

### **Output Requirements**
Return a JSON object containing:
- `"sim_score"` (int): A similarity score between 0-100, representing how similar the conversations themselves are in **trajectory** based on the SCDs.
- `"reason"` (string, ≤30 words): A brief explanation of why the score was given, referencing key conversational dynamics.

### **Output Format (JSON)**
```json
{{
    "sim_score": <int>,
    "reason": "<brief explanation (≤30 words)>"
}}

### **Conversations**
Conversation 1 SCD:
{pred}

Conversation 2 SCD:
{ref}
"""
    score, reason = get_gpt_compare_score(x, y, naive_gpt_compare_scd_prompt)
    return score, reason


def get_naive_gpt_compare_score_Transcripts(x, y):
    naive_gpt_compare_transcript_prompt = """Compare the following two online conversations and rate their similarity on a scale from 1 to 100, based on their trajectory.
    
### **Definition of Trajectory**
The trajectory of a conversation refers to its **dynamics**, including:
- **Changes in tone** (e.g., neutral to argumentative, formal to casual, sarcastic or sincere).
- **Patterns of interaction** (e.g., back-and-forth exchanges, long monologues, interruptions).
- **Conversation strategies** (e.g., persuasion, questioning, storytelling).
- **Order of the above trajectory events**

### **Ignore**:
- The topics discussed.
- Specific factual content.

### **Output Requirements**
Return a JSON object containing:
- `"sim_score"` (int): A similarity score between 0-100, representing how similar the conversations are in **trajectory**.
- `"reason"` (string, ≤30 words): A brief explanation of why the score was given, referencing key conversational dynamics.

### **Output Format (JSON)**
```json
{{
    "sim_score": <int>,
    "reason": "<brief explanation (≤30 words)>"
}}

### **Conversations**
Conversation 1:
{pred}

Conversation 2:
{ref}
"""
    score, reason = get_gpt_compare_score(x, y, naive_gpt_compare_transcript_prompt)
    return score, reason