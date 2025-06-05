import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import random
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config_loader import AppConfig
from utils.token_utils import decode_token, decode_token_path
from utils.ngram_utils import NgramUtil
from llm_interface.api_client import ApiClient
from llm_interface.chat_formatter import ChatTemplateFormatter
from classification.refusal_detector import RefusalDetector
from data_handling.database import DatabaseManager
from core.sampling import Sampler
from core.models import PromptData # Assuming PromptData is defined
from core.beam import Beam

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

        self.sampler = Sampler(asdict(config.sampling_beam_starters))
        
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
            if ngram_type_key.startswith("2"):
                ngram_size_str = "bigram"
            elif ngram_type_key.startswith("3"):
                ngram_size_str = "trigram"
            else:
                ngram_size_str = ngram_type_key
            for ngram_text in ngram_list:
                self.db_manager.update_ngram_quota(self.model_db_id, ngram_size_str, ngram_text, is_refusal)


    # core/explorer.py  –  replace the whole explore() method
    def explore(self) -> None:
        """
        Breadth-first beam expansion that matches the requested flow:

        • layer-0: prompt prefix, pick top-k next-tokens (after min_p etc.)
        • append `pilot_beam_length` greedy tokens to each child
        • run refusal detector on the resulting text
        • repeat for the next position, branching every live beam
        • keep at most `branch_cap` beams – stratified 50/50 by refusal state
        """
        cfg  = self.config
        ex   = cfg.exploration
        samp = self.sampler
        pilot_len   = ex.pilot_beam_length
        cap         = ex.branch_cap
        balance_p   = ex.balance_ratio

        # --- frontier initialisation ---------------------------------
        frontier: list[Beam] = [Beam(tokens_raw=[], decoded_words=[], depth=0)]

        while frontier:
            # -------- expand every live beam in parallel --------------
            next_frontier: list[Beam] = []

            # batch RPCs for speed – one /completions per beam
            reqs = []
            for beam in frontier:
                prefix_text = decode_token_path(beam.tokens_raw)
                full_prompt = (self.chat_formatter.build_prompt(self.prompt_data.text, prefix_text)
                            if self.chat_formatter
                            else self.prompt_data.text + prefix_text)
                reqs.append((beam, full_prompt))

            # Send requests concurrently but keep ordering
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                fut2beam = {pool.submit(
                    self.api_client.get_next_token_logprobs,
                    prompt_prefix=prompt,
                    top_n_logprobs = ex.beam_starters_top_k * 2,
                    sampling_params = asdict(cfg.sampling_beam_starters)
                ): beam for beam, prompt in reqs}

                for fut in as_completed(fut2beam):
                    beam = fut2beam[fut]
                    try:
                        lp_resp = fut.result()
                    except Exception as e:
                        logger.error(f"logprob RPC failed: {e}")
                        continue

                    cand_tok_raw = samp.sample_candidate_tokens(
                        lp_resp.logprobs,
                        ex.beam_starters_top_k
                    )
                    if not cand_tok_raw:
                        continue

                    # each candidate becomes a child beam
                    for tok_raw in cand_tok_raw:
                        # greedy tail
                        greedy_prefix = (self.chat_formatter.build_prompt(
                            self.prompt_data.text,
                            decode_token_path(beam.tokens_raw + [tok_raw])
                        ) if self.chat_formatter else
                            self.prompt_data.text + decode_token_path(beam.tokens_raw + [tok_raw]))

                        try:
                            greedy_resp = self.api_client.greedy_sample(
                                greedy_prefix,
                                num_tokens = pilot_len
                            )
                            tail_raw = greedy_resp.token_strings_raw
                        except Exception as e:
                            logger.error(f"greedy RPC failed: {e}")
                            tail_raw = []

                        full_raw_path = beam.tokens_raw + [tok_raw] + tail_raw
                        full_decoded  = decode_token_path(full_raw_path)

                        # refusal classification
                        is_ref, conf, lbl = self.refusal_detector.is_refusal(
                            self.prompt_data.text, full_decoded,
                            threshold=cfg.refusal_classifier.threshold
                        )

                        # record (exactly as old code did)
                        self.db_manager.record_exploration_result(
                            model_db_id = self.model_db_id,
                            prompt_db_id = self.prompt_data.db_id,
                            beam_path_raw = full_raw_path,
                            split_token_raw = tok_raw,
                            full_generation_text_decoded = full_decoded,
                            is_refusal = is_ref,
                            refusal_label = lbl,
                            refusal_confidence = conf,
                            exploration_depth = beam.depth
                        )

                        # bookkeeping + quotas
                        cleaned_words = (beam.decoded_words +
                                        self.ngram_util._tokenize_and_clean(
                                            decode_token(tok_raw)
                                        ) +
                                        self.ngram_util._tokenize_and_clean(
                                            decode_token_path(tail_raw)
                                        ))
                        self._update_all_quotas(
                            current_beam_path_raw = beam.tokens_raw,
                            split_token_raw       = tok_raw,
                            is_refusal            = is_ref,
                            previous_decoded_words_for_ngram = beam.decoded_words
                        )

                        # child beam for next layer – it starts *after* the pilot segment
                        next_frontier.append(
                            Beam(
                                tokens_raw   = full_raw_path,
                                decoded_words = cleaned_words,
                                depth = beam.depth + 1,
                                is_refusal = is_ref,
                                conf = conf
                            )
                        )

            # -------- branch-cap & balancing ---------------------------
            if len(next_frontier) > cap:
                refusals     = [b for b in next_frontier if  b.is_refusal]
                non_refusals = [b for b in next_frontier if not b.is_refusal]

                keep_ref = int(cap * balance_p)
                keep_non = cap - keep_ref

                random.shuffle(refusals)
                random.shuffle(non_refusals)

                next_frontier = refusals[:keep_ref] + non_refusals[:keep_non]

            # depth / length termination
            if not next_frontier:
                break
            if next_frontier[0].depth >= ex.max_exploration_depth:
                break
            if any(len(b.tokens_raw) >= ex.max_total_generation_length for b in next_frontier):
                break

            frontier = next_frontier
