from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class PromptData:
    original_id: str
    category: Optional[str]
    text: str
    db_id: Optional[int] = None # To be filled after inserting into DB

@dataclass
class APIClientConfig:
    base_url: str
    api_key: str
    model_name: str
    chat_template_model_id: Optional[str]
    timeout_seconds: int

@dataclass
class ExplorationResult:
    model_db_id: int
    prompt_db_id: int
    beam_path_raw_json: str # JSON string of list of raw tokens
    split_token_raw: str
    split_token_decoded: str
    full_generation_text_decoded: str
    is_refusal: bool
    refusal_label: Optional[str]
    refusal_confidence: Optional[float]
    timestamp: str # ISO format string

@dataclass
class APILogProbsResponse:
    """Holds data for a single token and its top logprobs from the API."""
    # For /v1/completions, this is usually a list of (token_str, logprob_float)
    # The structure might vary slightly based on API (e.g. vLLM vs OpenAI)
    # For simplicity, assuming a list of (token_string, log_probability)
    logprobs: List[tuple[str, float]] = field(default_factory=list)
    # If the API returns the chosen token separately or if we need to infer it
    chosen_token_string: Optional[str] = None 

@dataclass
class APIGreedySampleResponse:
    """Holds the sequence of greedily sampled tokens."""
    token_strings_raw: List[str] = field(default_factory=list)
    finish_reason: Optional[str] = None