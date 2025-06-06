import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import random
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent

from utils.config_loader import AppConfig
from utils.token_utils import decode_token, decode_token_path
from utils.ngram_utils import NgramUtil
from llm_interface.api_client import ApiClient
from llm_interface.chat_formatter import ChatTemplateFormatter
from classification.refusal_worker import RefusalWorker
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
        refusal_worker: RefusalWorker,
        db_manager: DatabaseManager,
        ngram_util: NgramUtil,
        model_db_id: int,
        prompt_data: PromptData,
    ):
        self.config = config
        self.api_client = api_client
        self.chat_formatter = chat_formatter
        self.refusal_worker = refusal_worker
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


    # core/explorer.py
    def explore(self) -> None:
        """
        Breadth-first beam search with two batched RPC phases per layer:

        1) /completions (logprobs=K)  – one request per live beam
        2) /completions (greedy tail) – one request per (beam × candidate)
        3) Refusal classification     – batched inside RefusalWorker
        """
        cfg        = self.config
        ex         = cfg.exploration
        sampler    = self.sampler
        pilot_len  = ex.pilot_beam_length
        cap        = ex.branch_cap
        balance_p  = ex.balance_ratio
        max_depth  = ex.max_split_positions

        frontier: list[Beam] = [Beam(tokens_raw=[], decoded_words=[], depth=0)]

        pool = ThreadPoolExecutor(max_workers=cfg.max_workers)
        try:
            while frontier:
                # ───────────────────────── phase 1 – logprobs ─────────────────────────
                fut_lp = {
                    pool.submit(
                        self.api_client.get_next_token_logprobs,
                        prompt_prefix=(
                            self.chat_formatter.build_prompt(
                                self.prompt_data.text,
                                decode_token_path(b.tokens_raw)
                            ) if self.chat_formatter else
                            self.prompt_data.text + decode_token_path(b.tokens_raw)
                        ),
                        top_n_logprobs=ex.beam_starters_top_k * 2,
                        sampling_params=asdict(cfg.sampling_beam_starters),
                    ): b
                    for b in frontier
                }

                beam_candidates: list[tuple[Beam, str, str]] = []          # (parent, split_tok_raw, greedy_prefix)

                for fut in as_completed(fut_lp):
                    beam = fut_lp[fut]
                    try:
                        lp_resp = fut.result()
                    except Exception as e:
                        logger.error(f"logprob RPC failed: {e}")
                        continue

                    cand_tokens = sampler.sample_candidate_tokens(
                        lp_resp.logprobs, ex.beam_starters_top_k
                    )
                    for tok_raw in cand_tokens:
                        greedy_prefix = (
                            self.chat_formatter.build_prompt(
                                self.prompt_data.text,
                                decode_token_path(beam.tokens_raw + [tok_raw])
                            ) if self.chat_formatter else
                            self.prompt_data.text + decode_token_path(beam.tokens_raw + [tok_raw])
                        )
                        beam_candidates.append((beam, tok_raw, greedy_prefix))

                if not beam_candidates:
                    break

                # ──────────────────────── phase 2 – greedy tails ───────────────────────
                fut_tail = {
                    pool.submit(
                        self.api_client.greedy_sample,
                        prompt_prefix=gp,
                        num_tokens=pilot_len,
                    ): (parent, tok_raw)
                    for parent, tok_raw, gp in beam_candidates
                }

                # collect detector futures for this search layer
                pending: list[
                    tuple[concurrent.futures.Future, Beam, str, list[str], str, list[str]]
                ] = []   # (future, parent, tok_raw, tail_raw, gen_text, parent_path_raw)

                next_frontier: list[Beam] = []

                for fut in as_completed(fut_tail):
                    parent, tok_raw = fut_tail[fut]
                    try:
                        tail_resp = fut.result()
                        tail_raw  = tail_resp.token_strings_raw
                    except Exception as e:
                        logger.error(f"greedy RPC failed: {e}")
                        tail_raw = []

                    path_raw = parent.tokens_raw + [tok_raw] + tail_raw
                    gen_text = decode_token_path(path_raw)

                    future = self.refusal_worker.submit(self.prompt_data.text, gen_text)
                    pending.append((future, parent, tok_raw, tail_raw, gen_text, path_raw))

                # ──────────────── wait for all refusal results together ────────────────
                for fut, parent, tok_raw, tail_raw, gen_text, path_raw in pending:
                    is_ref, conf, lbl = fut.result()

                    self.db_manager.record_exploration_result(
                        self.model_db_id,
                        self.prompt_data.db_id,
                        path_raw,
                        tok_raw,
                        gen_text,
                        is_ref,
                        lbl,
                        conf,
                        parent.depth,
                    )

                    self._update_all_quotas(
                        current_beam_path_raw            = parent.tokens_raw,
                        split_token_raw                  = tok_raw,
                        is_refusal                       = is_ref,
                        previous_decoded_words_for_ngram = parent.decoded_words,
                    )

                    child_dec_words = parent.decoded_words + \
                        self.ngram_util._tokenize_and_clean(decode_token(tok_raw))

                    next_frontier.append(
                        Beam(
                            tokens_raw    = parent.tokens_raw + [tok_raw],
                            decoded_words = child_dec_words,
                            depth         = parent.depth + 1,
                            is_refusal    = is_ref,
                            conf          = conf,
                        )
                    )

                # ──────────────── branch-capping / balancing ────────────────
                if len(next_frontier) > cap:
                    refusals     = [b for b in next_frontier if  b.is_refusal]
                    non_refusals = [b for b in next_frontier if not b.is_refusal]
                    keep_ref = int(cap * balance_p)
                    keep_non = cap - keep_ref
                    random.shuffle(refusals)
                    random.shuffle(non_refusals)
                    next_frontier = refusals[:keep_ref] + non_refusals[:keep_non]

                if not next_frontier or next_frontier[0].depth >= max_depth:
                    break

                frontier = next_frontier
        finally:
            pool.shutdown(wait=True)
            self.refusal_worker.shutdown()


