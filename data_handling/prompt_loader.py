import json
import logging
from typing import Iterator, Tuple, Optional, Dict, Any
from pathlib import Path

from datasets import load_dataset, Dataset # For type hinting
from core.models import PromptData

logger = logging.getLogger(__name__)

class PromptLoader:
    def __init__(self, dataset_config: Dict[str, Any]):
        """
        Initializes the PromptLoader with a dataset configuration.
        dataset_config is one item from the `datasets` list in the main config.
        """
        self.config = dataset_config
        self.name = dataset_config['name']
        self.type = dataset_config['type']
        
        # Fields for JSONL, with defaults
        self.prompt_field = dataset_config.get('prompt_field', 'question')
        self.category_field = dataset_config.get('category_field', 'category')
        self.id_field = dataset_config.get('id_field', 'id')

        logger.info(f"PromptLoader initialized for dataset '{self.name}' of type '{self.type}'.")

    def _load_from_jsonl(self) -> Iterator[PromptData]:
        jsonl_path_str = self.config.get('path')
        if not jsonl_path_str:
            logger.error(f"Path not specified for JSONL dataset '{self.name}'.")
            return
        
        jsonl_path = Path(jsonl_path_str)
        if not jsonl_path.exists():
            logger.error(f"JSONL file not found for dataset '{self.name}': {jsonl_path}")
            return

        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        prompt_text = data.get(self.prompt_field)
                        original_id = str(data.get(self.id_field, f"line_{i+1}")) # Default ID if missing
                        category = data.get(self.category_field)

                        if not prompt_text:
                            logger.warning(f"Skipping line {i+1} in {jsonl_path}: missing prompt field '{self.prompt_field}'.")
                            continue
                        
                        yield PromptData(original_id=original_id, category=category, text=prompt_text)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line {i+1} in {jsonl_path}.")
                    except Exception as e:
                        logger.warning(f"Error processing line {i+1} in {jsonl_path}: {e}")
        except IOError as e:
            logger.error(f"Could not read JSONL file {jsonl_path}: {e}")


    def _load_from_hf_dataset(self) -> Iterator[PromptData]:
        hf_id = self.config.get('hf_id')
        hf_split = self.config.get('hf_split', 'train')
        if not hf_id:
            logger.error(f"Hugging Face ID (hf_id) not specified for dataset '{self.name}'.")
            return

        try:
            logger.info(f"Loading Hugging Face dataset '{hf_id}', split '{hf_split}'. This may take time...")
            # Suppress verbose logging from datasets library during load
            datasets_log_level = logging.getLogger("datasets").level
            hfh_log_level = logging.getLogger("huggingface_hub").level
            logging.getLogger("datasets").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            
            dataset: Dataset = load_dataset(hf_id, split=hf_split, trust_remote_code=True) # Added trust_remote_code
            
            logging.getLogger("datasets").setLevel(datasets_log_level)
            logging.getLogger("huggingface_hub").setLevel(hfh_log_level)
            logger.info(f"Dataset '{hf_id}' loaded. Processing items...")

        except Exception as e:
            logger.error(f"Failed to load Hugging Face dataset {hf_id}: {e}", exc_info=True)
            return

        for i, item in enumerate(dataset):
            prompt_text: Optional[str] = None
            category: Optional[str] = None # ShareGPT usually doesn't have explicit category per prompt
            original_id: str = f"{hf_id}_{hf_split}_{i}"

            # Attempt to extract prompt from ShareGPT-like structures
            if "conversations" in item and isinstance(item["conversations"], list):
                for turn in item["conversations"]:
                    if isinstance(turn, dict) and turn.get("from", "").lower() == "human" and "value" in turn:
                        prompt_text = str(turn["value"])
                        break # Take the first human turn
            elif "prompt" in item and isinstance(item["prompt"], str):
                prompt_text = item["prompt"]
            elif "text" in item and isinstance(item["text"], str): # Generic fallback
                # Avoid taking assistant responses as prompts if possible
                if not any(kw in item["text"].lower() for kw in ["assistant:", "bot:", "gpt:", "\n\nassistant", "human:"]):
                     prompt_text = item["text"]
            
            # Try to get category if available (less common in ShareGPT)
            if "category" in item and isinstance(item["category"], str):
                category = item["category"]
            elif self.config.get("hf_category_field") and self.config["hf_category_field"] in item:
                 category = str(item[self.config["hf_category_field"]])


            if prompt_text:
                yield PromptData(original_id=original_id, category=category, text=prompt_text.strip())
            else:
                logger.warning(f"Could not extract a suitable prompt from item {i} in HF dataset '{hf_id}'. Item keys: {list(item.keys())}")


    def load_prompts(self) -> Iterator[PromptData]:
        if self.type == "jsonl":
            yield from self._load_from_jsonl()
        elif self.type == "hf_dataset":
            yield from self._load_from_hf_dataset()
        else:
            logger.error(f"Unsupported dataset type '{self.type}' for dataset '{self.name}'.")
            return