import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Small config records – all simple dataclasses                              #
# --------------------------------------------------------------------------- #
@dataclass
class APIConfig:
    base_url: str
    api_key: str
    model_name: str
    chat_template_model_id: Optional[str] = ""
    timeout_seconds: int = 120


@dataclass
class ExplorationConfig:
    beam_starters_top_k: int
    pilot_beam_length: int
    max_split_positions: int
    branch_cap: int
    balance_ratio: float



@dataclass
class SamplingBeamStartersConfig:
    temperature: float
    min_p: float
    top_p: Optional[float] = None
    top_k_filter: Optional[int] = None


@dataclass
class DatasetConfig:
    name: str
    type: str
    path: Optional[str] = None
    hf_id: Optional[str] = None
    hf_split: Optional[str] = None
    prompt_field: str = "question"
    category_field: str = "category"
    id_field: str = "id"


@dataclass
class QuotasConfig:
    single_token: int
    token_pair: int
    bigram: int
    trigram: int


@dataclass
class NgramsConfig:
    language: str


@dataclass
class RefusalClassifierConfig:
    model_id: str
    threshold: float


# --------------------------------------------------------------------------- #
#  Top-level wrapper – uses the small records above                           #
# --------------------------------------------------------------------------- #
class AppConfig:
    def __init__(self, data: Dict[str, Any]):
        self.api: APIConfig = APIConfig(**data["api"])
        self.exploration: ExplorationConfig = ExplorationConfig(**data["exploration"])
        self.sampling_beam_starters: SamplingBeamStartersConfig = (
            SamplingBeamStartersConfig(**data["sampling_beam_starters"])
        )
        self.datasets: List[DatasetConfig] = [
            DatasetConfig(**d) for d in data["datasets"]
        ]
        self.quotas: QuotasConfig = QuotasConfig(**data["quotas"])
        self.ngrams: NgramsConfig = NgramsConfig(**data["ngrams"])
        self.refusal_classifier: RefusalClassifierConfig = RefusalClassifierConfig(
            **data["refusal_classifier"]
        )
        self.database_path: str = data["database_path"]
        self.logging_level: str = data["logging_level"]
        self.max_workers: int = data.get("max_workers", 1)  # default fallback


# --------------------------------------------------------------------------- #
#  Loader helper                                                              #
# --------------------------------------------------------------------------- #
def load_config(config_path: str) -> AppConfig:
    """
    Parse a YAML config file into an AppConfig instance.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return AppConfig(cfg)
    except FileNotFoundError:
        logger.error(f"Config file '{config_path}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in '{config_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading or validating config '{config_path}': {e}")
        raise


# --------------------------------------------------------------------------- #
#  Convenience: allow dict-style access (obj["field"]) on all records         #
# --------------------------------------------------------------------------- #
def _make_subscriptable(cls):
    def __getitem__(self, key):
        return getattr(self, key)
    cls.__getitem__ = __getitem__
    return cls


for _cls in (
    APIConfig,
    ExplorationConfig,
    SamplingBeamStartersConfig,
    DatasetConfig,
    QuotasConfig,
    NgramsConfig,
    RefusalClassifierConfig,
    AppConfig,
):
    _make_subscriptable(_cls)
