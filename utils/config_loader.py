import yaml
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Basic Pydantic-like structures for validation (can be expanded)
class APIConfig:
    base_url: str
    api_key: str
    model_name: str
    chat_template_model_id: str
    timeout_seconds: int

class ExplorationConfig:
    beam_starters_top_k: int
    greedy_sample_length: int
    max_exploration_depth: int
    max_total_generation_length: int

class SamplingBeamStartersConfig:
    temperature: float
    min_p: float
    top_p: Optional[float] = None
    top_k_filter: Optional[int] = None

class DatasetConfig:
    name: str
    type: str
    path: Optional[str] = None
    hf_id: Optional[str] = None
    hf_split: Optional[str] = None
    prompt_field: Optional[str] = "question"
    category_field: Optional[str] = "category"
    id_field: Optional[str] = "id"


class QuotasConfig:
    single_token: int
    token_pair: int
    bigram: int
    trigram: int

class NgramsConfig:
    language: str

class RefusalClassifierConfig:
    model_id: str
    threshold: float

class AppConfig:
    api: APIConfig
    exploration: ExplorationConfig
    sampling_beam_starters: SamplingBeamStartersConfig
    datasets: List[DatasetConfig]
    quotas: QuotasConfig
    ngrams: NgramsConfig
    refusal_classifier: RefusalClassifierConfig
    database_path: str
    logging_level: str
    max_workers: int

    def __init__(self, data: Dict[str, Any]):
        self.api = APIConfig(**data['api'])
        self.exploration = ExplorationConfig(**data['exploration'])
        self.sampling_beam_starters = SamplingBeamStartersConfig(**data['sampling_beam_starters'])
        self.datasets = [DatasetConfig(**ds_data) for ds_data in data['datasets']]
        self.quotas = QuotasConfig(**data['quotas'])
        self.ngrams = NgramsConfig(**data['ngrams'])
        self.refusal_classifier = RefusalClassifierConfig(**data['refusal_classifier'])
        self.database_path = data['database_path']
        self.logging_level = data['logging_level']
        self.max_workers = data.get('max_workers', 1) # Default to 1 if not specified

def load_config(config_path: str) -> AppConfig:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return AppConfig(config_data)
    except FileNotFoundError:
        logger.error(f"Config file '{config_path}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in '{config_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading or validating config '{config_path}': {e}")
        raise

# Helper to make Pydantic-like classes subscriptable
def _make_subscriptable(cls):
    def getitem(self, key):
        return getattr(self, key)
    cls.__getitem__ = getitem
    return cls

APIConfig = _make_subscriptable(APIConfig)
ExplorationConfig = _make_subscriptable(ExplorationConfig)
SamplingBeamStartersConfig = _make_subscriptable(SamplingBeamStartersConfig)
DatasetConfig = _make_subscriptable(DatasetConfig)
QuotasConfig = _make_subscriptable(QuotasConfig)
NgramsConfig = _make_subscriptable(NgramsConfig)
RefusalClassifierConfig = _make_subscriptable(RefusalClassifierConfig)
AppConfig = _make_subscriptable(AppConfig)