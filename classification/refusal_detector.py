# classification/refusal_detector.py
# Adapted from antislop-vllm/utils/refusal_detector.py
from __future__ import annotations

import sys
import threading
import logging
from typing import Tuple, Optional, Dict

import torch                                   # add these two lines
def _noop_compile(fn=None, *a, **kw):
    """
    Behaves like torch.compile, but just returns the
    original function so nothing is compiled.

    • If used as @torch.compile(dynamic=True) → returns
      a decorator waiting for the real fn.
    • If used as @torch.compile → gets fn directly.
    """
    if fn is None:                       # called with kwargs only
        def decorator(real_fn):
            return real_fn
        return decorator
    return fn                            # called with the fn itself

if hasattr(torch, "compile"):
    torch.compile = _noop_compile        # monkey-patch once
    
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)

# Tunables from antislop-vllm
MAX_LEN          = 256
MIN_USER_TOKENS  = 40
MAX_USER_TOKENS  = 100

class RefusalDetector:
    _instances: Dict[str, "RefusalDetector"] = {} # Cache per model_id
    _lock = threading.Lock()

    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_len: int = MAX_LEN,
        use_bnb_quantization: bool = True, # Added flag
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_len = max_len
        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}
        self._failed = False
        self._error: Optional[str] = None
        self._tok_lock = threading.RLock() # For thread-safe tokenizer usage

        try:
            logger.info(f"Loading refusal detector tokenizer: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer.model_max_length = max_len
            if self.tokenizer.is_fast:
                self.tokenizer._tokenizer.enable_truncation(max_length=max_len)

            logger.info(f"Loading refusal detector model: {model_id} on device: {device}")
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            for attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, "eager")

            model_kwargs = {
                "config": cfg, # Pass modified config
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                # "attn_implementation": "eager", # Already set in cfg
                "trust_remote_code": True,
            }

            if use_bnb_quantization and self.device == "cuda":
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    logger.info("Using BitsAndBytes 4-bit quantization for refusal detector.")
                except ImportError:
                    logger.warning("BitsAndBytes not installed. Quantization for refusal detector disabled. pip install bitsandbytes")
                except Exception as e:
                    logger.warning(f"Failed to set up BitsAndBytes quantization: {e}. Proceeding without quantization.")


            try:                                            # ModernBERT & friends
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    reference_compile=False,                # <- disable torch.compile wrappers
                    **model_kwargs
                )
            except TypeError:                               # models that don’t recognise the arg
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    **model_kwargs
                )
            # No explicit .to(device) needed if quantization_config is used with BitsAndBytes,
            # as it handles device placement. If not using BNB, model needs .to(device).
            if not (use_bnb_quantization and self.device == "cuda" and "quantization_config" in model_kwargs):
                 self.model.to(self.device)

            self.model.eval() # Set to evaluation mode

            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            logger.info(f"Refusal detector '{model_id}' loaded successfully.")

        except Exception as exc:
            self._failed = True
            self._error = str(exc)
            logger.error(f"Failed to load refusal detector '{model_id}': {exc}", exc_info=True)
            # Print to stderr as well for high visibility during script runs
            print(f"\n[RefusalDetector ERROR] Failed to load '{model_id}': {exc}\n", file=sys.stderr, flush=True)


    @classmethod
    def get(
        cls,
        model_id: str = "NousResearch/Minos-v1",
        device: Optional[str] = None,
        max_len: int = MAX_LEN,
        use_bnb_quantization: bool = True,
    ) -> "RefusalDetector":
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_key = f"{model_id}_{resolved_device}_{max_len}_{use_bnb_quantization}"

        with cls._lock:
            if cache_key not in cls._instances:
                logger.info(f"Creating new RefusalDetector instance for key: {cache_key}")
                cls._instances[cache_key] = RefusalDetector(
                    model_id, device=resolved_device, max_len=max_len, use_bnb_quantization=use_bnb_quantization
                )
            else:
                logger.debug(f"Reusing cached RefusalDetector instance for key: {cache_key}")
        return cls._instances[cache_key]

    @staticmethod
    def _chat_wrap(user: str, assistant: str) -> str:
        # This format is specific to NousResearch/Minos-v1
        # Adjust if using a different classifier model
        return f"<|user|>\n{user}\n<|assistant|>\n{assistant}"

    def _truncate_inputs(
        self,
        user_text: str,
        assistant_text: str,
    ) -> tuple[str, str]:
        if not self.tokenizer: return user_text, assistant_text # Should not happen if init checks

        with self._tok_lock:
            user_ids = self.tokenizer.encode(user_text, add_special_tokens=False)[:MAX_USER_TOKENS]
            # Decode to ensure text matches tokenization, then re-encode for length check
            temp_user_text = self.tokenizer.decode(user_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            assistant_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)
            static_ids = self.tokenizer.encode(self._chat_wrap("", ""), add_special_tokens=False) # Template overhead

            current_user_len = len(self.tokenizer.encode(temp_user_text, add_special_tokens=False))
        total_len = len(static_ids) + current_user_len + len(assistant_ids)

        if total_len <= self.max_len:
            return temp_user_text, assistant_text

        # Reduce user_text if it's too long and causing overflow
        excess = total_len - self.max_len
        
        # How many user tokens can we drop while respecting MIN_USER_TOKENS?
        # We need to re-encode user_text to get its current token length for this calculation
        
        can_drop_from_user = max(0, current_user_len - MIN_USER_TOKENS)
        drop_amount = min(excess, can_drop_from_user)

        if drop_amount > 0:
            with self._tok_lock:
                final_user_ids = self.tokenizer.encode(temp_user_text, add_special_tokens=False)[:-drop_amount]
                user_text = self.tokenizer.decode(final_user_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            user_text = temp_user_text # Use the initially truncated user_text

        # Any remaining excess will be handled by the tokenizer's truncation on the combined input.
        return user_text, assistant_text


    def classify(
        self,
        user_text: str,
        assistant_text: str,
    ) -> Tuple[str, int, float]:
        if self._failed or not self.model or not self.tokenizer:
            logger.warning(f"RefusalDetector for '{self.model_id}' is not available or failed to load. Returning sentinel values.")
            return "error", -1, 0.0

        user_text_processed, assistant_text_processed = self._truncate_inputs(user_text, assistant_text)
        combined_text = self._chat_wrap(user_text_processed, assistant_text_processed)

        try:
            with self._tok_lock: # Ensure thread-safe tokenization
                inputs = self.tokenizer(
                    combined_text,
                    return_tensors="pt",
                    padding=False, # Let tokenizer handle if needed, or set to True
                    truncation=True,
                    max_length=self.max_len,
                ).to(self.device)

            with torch.no_grad():
                # ModernBERT inside Minos tries to run under torch.compile; wrap in
                # a “no-opt” context so FX-on-Dynamo tracing never triggers.
                try:
                    noopt = torch._dynamo.noopt_context   # available in 2.1+
                except AttributeError:                    # fall-back for older Torch
                    from contextlib import nullcontext
                    noopt = nullcontext
                with noopt():
                    logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                label_id = int(torch.argmax(probs).item())
                confidence = float(probs[label_id].item()) # Ensure it's a Python float

            label_text = self.id2label.get(label_id, str(label_id))
            
            # Clean up tensors
            del inputs, logits, probs
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return label_text, label_id, confidence

        except Exception as e:
            logger.error(f"Error during refusal classification for '{self.model_id}': {e}", exc_info=True)
            return "error", -1, 0.0


    def is_refusal(
        self,
        user_text: str,
        assistant_text: str,
        threshold: float = 0.8,
    ) -> Tuple[bool, float, str]:
        if self._failed:
            return False, 0.0, "error" # Sentinel for load failure

        label_text, _, confidence = self.classify(user_text, assistant_text)
        if label_text == "error":
            return False, 0.0, "error" # Sentinel for classification error

        is_ref = (label_text.lower().strip() == "refusal") and (confidence >= threshold)
        logger.debug(f"Refusal check: Label='{label_text}', Confidence={confidence:.4f}, Threshold={threshold}, IsRefusal={is_ref}")
        return is_ref, confidence, label_text