from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# --------------------------------------------------------------------------- #
#  Prompt metadata handed around the pipeline                                 #
# --------------------------------------------------------------------------- #
@dataclass
class PromptData:
    original_id: str
    category: Optional[str]
    text: str
    db_id: Optional[int] = None
    # dataset the prompt came from; not always present when first loaded
    source_dataset_name: Optional[str] = None


# --------------------------------------------------------------------------- #
#  API client / config helpers                                                #
# --------------------------------------------------------------------------- #
@dataclass
class APIClientConfig:
    base_url: str
    api_key: str
    model_name: str
    chat_template_model_id: Optional[str]
    timeout_seconds: int


# --------------------------------------------------------------------------- #
#  Records persisted to SQLite                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ExplorationResult:
    model_db_id: int
    prompt_db_id: int
    beam_path_raw_json: str
    split_token_raw: str
    split_token_decoded: str
    full_generation_text_decoded: str
    is_refusal: bool
    refusal_label: Optional[str]
    refusal_confidence: Optional[float]
    timestamp: str


# --------------------------------------------------------------------------- #
#  LLM-API response holders                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class APILogProbsResponse:
    logprobs: List[tuple[str, float]] = field(default_factory=list)
    chosen_token_string: Optional[str] = None


@dataclass
class APIGreedySampleResponse:
    token_strings_raw: List[str] = field(default_factory=list)
    finish_reason: Optional[str] = None
