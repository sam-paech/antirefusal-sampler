import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from utils.config_loader import AppConfig
from utils.token_utils import decode_token, decode_token_path
from utils.ngram_utils import NgramUtil
from llm_interface.api_client import ApiClient
from llm_interface.chat_formatter import ChatTemplateFormatter
from classification.refusal_detector import RefusalDetector
from data_handling.database import DatabaseManager
from core.sampling import Sampler
from core.models import PromptData # Assuming PromptData is defined

logger = logging.getLogger(__name__)

class BeamExplorer:
    def __init__(
        self,
        config: AppConfig,
        api_client: ApiClient,
        chat_formatter: Optional[ChatTemplateFormatter],
        refusal_detector: RefusalDetector,
        db_manager: DatabaseManager,
        ngram_util: NgramUtil,
        model_db_id: int,
        prompt_data: PromptData,
    ):
        self.config = config
        self.api_client = api_client
        self.chat_formatter = chat_formatter
        self.refusal_detector = refusal_detector
        self.db_manager = db_manager
        self.ngram_util = ngram_util
        self.model_db_id = model_db_id
        self.prompt_data = prompt_data # Contains original_id, category, text, db_id

        self.sampler = Sampler(config.sampling_beam_starters)
        
        self.exploration_params = config.exploration
        self.quota_params = config.quotas

        # To store decoded words of the current path for n-gram generation
        self._current_path_decoded_words: List[str] = []


    def _is_quota_met(self, item_type: str, item_key_dict: Dict[str, Any], max_quota: int) -> bool:
        refusal_count, non_refusal_count = self.db_manager.get_quota_status(
            self.model_db_id, item_type, item_key_dict
        )
        total_count = refusal_count + non_refusal_count
        is_met = total_count >= max_quota
        if is_met:
            logger.debug(f"Quota met for {item_type} {item_key_dict}: {total_count}/{max_quota}")
        return is_met

    def _check_all_quotas_for_token(self, current_beam_path_raw: List[str], starter_token_raw: str) -> bool:
        """Checks if quotas are met for the starter_token and related n-grams/pairs."""
        # 1. Single token quota
        if self._is_quota_met('single_token', {'token_raw': starter_token_raw}, self.quota_params.single_token):
            return True # Quota met

        # 2. Token pair quota (if applicable)
        if current_beam_path_raw:
            prev_token_raw = current_beam_path_raw[-1]
            if self._is_quota_met('token_pair', {'prev_token_raw': prev_token_raw, 'split_token_raw': starter_token_raw}, self.quota_params.token_pair):
                return True

        # 3. N-gram quotas
        # Need the decoded words of the path *before* this starter_token, and the decoded starter_token
        # This requires careful management of `self._current_path_decoded_words`
        
        # Get decoded words for the path leading up to the current split point
        # This assumes `self._current_path_decoded_words` reflects the path *before* `starter_token_raw`
        # This state needs to be passed down or managed carefully in recursive calls.
        # For now, let's assume it's available.
        
        # This is tricky: self._current_path_decoded_words should be the state *before* adding starter_token.
        # The recursive call needs to update it.
        # Let's pass the previous_decoded_words_for_ngram explicitly to this check.
        
        # This part needs to be called from within the explore loop where previous_decoded_words is known.
        # For now, this function is a placeholder for the logic.
        # The actual call to ngram_util.get_relevant_ngrams_for_token will happen in the explore loop.
        
        # Placeholder: actual n-gram quota check will be more involved.
        # starter_token_decoded = decode_token(starter_token_raw)
        # relevant_ngrams = self.ngram_util.get_relevant_ngrams_for_token(
        #     self._current_path_decoded_words, # This needs to be the path *before* starter_token
        #     starter_token_decoded,
        #     [2, 3] # bigrams and trigrams
        # )
        # for ngram_type_key, ngram_list in relevant_ngrams.items(): # e.g., "bigram": ["word1 word2"]
        #     ngram_size_str = ngram_type_key # "bigram" or "trigram"
        #     quota_val = getattr(self.quota_params, ngram_size_str, 0)
        #     for ngram_text in ngram_list:
        #         if self._is_quota_met(ngram_size_str, {'ngram_text': ngram_text}, quota_val):
        #             return True
        
        return False # No quotas met that would stop exploration for this token

    def _update_all_quotas(self, current_beam_path_raw: List[str], split_token_raw: str, is_refusal: bool, previous_decoded_words_for_ngram: List[str]):
        # 1. Single token
        self.db_manager.update_token_quota(self.model_db_id, split_token_raw, is_refusal)

        # 2. Token pair
        if current_beam_path_raw: # current_beam_path_raw is path *before* split_token_raw
            prev_token_raw = current_beam_path_raw[-1]
            self.db_manager.update_token_pair_quota(self.model_db_id, prev_token_raw, split_token_raw, is_refusal)
        
        # 3. N-grams
        split_token_decoded = decode_token(split_token_raw)
        relevant_ngrams = self.ngram_util.get_relevant_ngrams_for_token(
            previous_decoded_words_for_ngram, # Decoded words of path *before* split_token
            split_token_decoded,
            [2, 3] # bigrams and trigrams
        )
        for ngram_type_key, ngram_list in relevant_ngrams.items():
            ngram_size_str = ngram_type_key # "bigram" or "trigram"
            for ngram_text in ngram_list:
                self.db_manager.update_ngram_quota(self.model_db_id, ngram_size_str, ngram_text, is_refusal)


    def explore(self, current_beam_path_raw: List[str], current_depth: int, current_path_decoded_words: List[str]):
        """
        Recursively explores token paths.
        current_beam_path_raw: List of raw token strings from the API forming the current path.
        current_depth: The number of branching steps taken so far.
        current_path_decoded_words: List of cleaned, decoded words corresponding to current_beam_path_raw.
        """
        logger.info(f"Exploring prompt '{self.prompt_data.original_id}', depth {current_depth}, path_len {len(current_beam_path_raw)}")

        # Base Cases for Recursion
        if current_depth >= self.exploration_params.max_exploration_depth:
            logger.debug(f"Max exploration depth ({self.exploration_params.max_exploration_depth}) reached.")
            return
        if len(current_beam_path_raw) >= self.exploration_params.max_total_generation_length:
            logger.debug(f"Max total generation length ({self.exploration_params.max_total_generation_length}) reached.")
            return

        # Construct Current Prefix for API
        current_path_decoded_text = decode_token_path(current_beam_path_raw)
        if self.chat_formatter:
            api_prompt_prefix = self.chat_formatter.build_prompt(self.prompt_data.text, current_path_decoded_text)
        else:
            api_prompt_prefix = self.prompt_data.text + current_path_decoded_text
        
        # Get Logprobs for the next token
        try:
            logprobs_response = self.api_client.get_next_token_logprobs(
                api_prompt_prefix,
                top_n_logprobs=self.config.api.get('top_logprobs_count_for_beams', self.exploration_params.beam_starters_top_k * 2), # Request more than needed
                sampling_params=self.config.sampling_beam_starters # Pass the dict directly
            )
        except Exception as e:
            logger.error(f"API call for logprobs failed at depth {current_depth} for prompt '{self.prompt_data.original_id}': {e}", exc_info=True)
            return

        if not logprobs_response.logprobs:
            logger.warning(f"No logprobs received from API at depth {current_depth} for prompt '{self.prompt_data.original_id}'. Stopping this path.")
            return

        # Select Candidate Beam Starter Tokens
        candidate_starter_tokens_raw = self.sampler.sample_candidate_tokens(
            logprobs_response.logprobs,
            self.exploration_params.beam_starters_top_k
        )

        if not candidate_starter_tokens_raw:
            logger.info(f"No candidate starter tokens after sampling at depth {current_depth} for prompt '{self.prompt_data.original_id}'.")
            return

        logger.debug(f"Depth {current_depth}: Candidate starter tokens: {candidate_starter_tokens_raw}")

        for starter_token_raw in candidate_starter_tokens_raw:
            # Quota Check (more detailed logic needed here)
            # This check needs the `current_path_decoded_words` to correctly form n-grams for quota checking.
            starter_token_decoded_for_ngram = decode_token(starter_token_raw)
            cleaned_starter_token_words = self.ngram_util._tokenize_and_clean(starter_token_decoded_for_ngram)

            # Check single token quota
            if self._is_quota_met('single_token', {'token_raw': starter_token_raw}, self.quota_params.single_token):
                continue

            # Check token pair quota
            if current_beam_path_raw:
                prev_token_raw = current_beam_path_raw[-1]
                if self._is_quota_met('token_pair', {'prev_token_raw': prev_token_raw, 'split_token_raw': starter_token_raw}, self.quota_params.token_pair):
                    continue
            
            # Check N-gram quotas
            # `current_path_decoded_words` is the list of cleaned words for the path *before* this starter_token
            relevant_ngrams_for_quota = self.ngram_util.get_relevant_ngrams_for_token(
                current_path_decoded_words, 
                starter_token_decoded_for_ngram, # Use the full decoded string for context
                [2, 3]
            )
            ngram_quota_met_flag = False
            for ngram_type_key, ngram_list in relevant_ngrams_for_quota.items():
                ngram_size_str = ngram_type_key 
                quota_val = getattr(self.quota_params, ngram_size_str.replace("gram",""), 0) # e.g. quotas.bigram
                if not quota_val: continue # Should not happen if config is right

                for ngram_text in ngram_list:
                    if self._is_quota_met(ngram_size_str, {'ngram_text': ngram_text}, quota_val):
                        ngram_quota_met_flag = True
                        break
                if ngram_quota_met_flag: break
            if ngram_quota_met_flag:
                continue


            # If all quotas allow, proceed with this starter_token_raw
            logger.debug(f"Depth {current_depth}: Processing starter token '{starter_token_raw}' (decoded: '{decode_token(starter_token_raw)}')")

            # Construct prefix for greedy sampling
            path_with_starter_decoded = decode_token_path(current_beam_path_raw + [starter_token_raw])
            if self.chat_formatter:
                greedy_sample_prefix = self.chat_formatter.build_prompt(self.prompt_data.text, path_with_starter_decoded)
            else:
                greedy_sample_prefix = self.prompt_data.text + path_with_starter_decoded
            
            # Greedy Sample Continuation
            try:
                greedy_response = self.api_client.greedy_sample(
                    greedy_sample_prefix,
                    self.exploration_params.greedy_sample_length
                )
                continuation_tokens_raw = greedy_response.token_strings_raw
            except Exception as e:
                logger.error(f"API call for greedy sampling failed for starter '{starter_token_raw}': {e}", exc_info=True)
                continue # Skip this starter token

            # Full Generation Path
            full_generated_path_raw = current_beam_path_raw + [starter_token_raw] + continuation_tokens_raw
            full_generated_text_decoded = decode_token_path(full_generated_path_raw)
            
            # Classify Refusal
            is_refusal, confidence, label = self.refusal_detector.is_refusal(
                self.prompt_data.text, # Original prompt
                full_generated_text_decoded, # Full decoded generation from this beam
                threshold=self.config.refusal_classifier.threshold
            )
            logger.info(f"Prompt '{self.prompt_data.original_id}', Starter '{decode_token(starter_token_raw)}', Depth {current_depth}: Refusal={is_refusal}, Label='{label}', Conf={confidence:.2f}")

            # Record Result
            self.db_manager.record_exploration_result(
                model_db_id=self.model_db_id,
                prompt_db_id=self.prompt_data.db_id,
                beam_path_raw=current_beam_path_raw + [starter_token_raw], # Path up to and including the split token
                split_token_raw=starter_token_raw,
                full_generation_text_decoded=full_generated_text_decoded,
                is_refusal=is_refusal,
                refusal_label=label,
                refusal_confidence=confidence,
                exploration_depth=current_depth 
            )

            # Update Quotas
            # `current_path_decoded_words` is correct here as it's the path *before* `starter_token_raw`
            self._update_all_quotas(current_beam_path_raw, starter_token_raw, is_refusal, current_path_decoded_words)

            # Recurse
            # Update current_path_decoded_words for the next level
            new_path_decoded_words = current_path_decoded_words + cleaned_starter_token_words
            
            self.explore(
                current_beam_path_raw=current_beam_path_raw + [starter_token_raw],
                current_depth=current_depth + 1,
                current_path_decoded_words=new_path_decoded_words
            )