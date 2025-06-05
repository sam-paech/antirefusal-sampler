import logging
import math
import random
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

class Sampler:
    def __init__(self, sampling_config: Dict[str, Any]):
        """
        Initializes the Sampler with sampling parameters.
        sampling_config is config.sampling_beam_starters
        """
        self.temperature = max(sampling_config.get('temperature', 1.0), 1e-6) # Avoid division by zero
        self.min_p = sampling_config.get('min_p')
        self.top_p = sampling_config.get('top_p')
        self.top_k_filter = sampling_config.get('top_k_filter') # This is for filtering before selection
        logger.info(
            f"Sampler initialized with: temp={self.temperature}, min_p={self.min_p}, "
            f"top_p={self.top_p}, top_k_filter={self.top_k_filter}"
        )

    def _apply_temperature(self, logprobs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        # Convert logprobs to probabilities, apply temperature, then normalize
        # exp(logprob / temp)
        # For numerical stability, subtract max_logprob first: exp(logprob - max_logprob / temp)
        if not logprobs:
            return []

        raw_logprobs_vals = [lp_val for _, lp_val in logprobs]
        max_logprob = max(raw_logprobs_vals) if raw_logprobs_vals else 0

        tempered_probs = []
        for token_str, lp_val in logprobs:
            # Subtract max_logprob before dividing by temperature to prevent overflow with exp
            # and underflow if lp_val is very small.
            scaled_logprob = (lp_val - max_logprob) / self.temperature
            try:
                prob = math.exp(scaled_logprob)
                tempered_probs.append((token_str, prob))
            except OverflowError:
                # If still overflows, this token had extremely low probability relative to max after temp
                # Or temperature is extremely small and lp_val is not max_logprob
                tempered_probs.append((token_str, 0.0))


        # Normalize probabilities
        total_prob_sum = sum(p for _, p in tempered_probs)
        if total_prob_sum == 0: # All probs became 0, distribute uniformly (should be rare)
            if not tempered_probs: return []
            uniform_prob = 1.0 / len(tempered_probs)
            return [(token_str, uniform_prob) for token_str, _ in tempered_probs]

        normalized_probs = [(token_str, p / total_prob_sum) for token_str, p in tempered_probs]
        return normalized_probs

    def _apply_min_p_filter(self, probs_dist: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if self.min_p is None or not probs_dist:
            return probs_dist

        max_prob = 0.0
        for _, p_val in probs_dist:
            if p_val > max_prob:
                max_prob = p_val
        
        if max_prob == 0: # Should not happen if probs_dist is not empty and normalized
            return probs_dist

        min_p_threshold = max_prob * self.min_p
        filtered_probs = [(token_str, p_val) for token_str, p_val in probs_dist if p_val >= min_p_threshold]

        if not filtered_probs: # If all tokens are filtered out, keep the one with highest original prob
            # This can happen if min_p is high. Fallback to the single best token.
            # Find the token with original max_prob from probs_dist
            best_fallback = max(probs_dist, key=lambda item: item[1], default=None)
            if best_fallback:
                return [best_fallback]
            return []


        # Re-normalize
        total_prob_sum = sum(p for _, p in filtered_probs)
        if total_prob_sum == 0: # Should be rare after filtering
             if not filtered_probs: return []
             uniform_prob = 1.0 / len(filtered_probs)
             return [(token_str, uniform_prob) for token_str, _ in filtered_probs]

        return [(token_str, p / total_prob_sum) for token_str, p in filtered_probs]

    def _apply_top_p_filter(self, probs_dist: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if self.top_p is None or not probs_dist:
            return probs_dist

        # Sort by probability in descending order
        sorted_probs = sorted(probs_dist, key=lambda item: item[1], reverse=True)

        cumulative_prob = 0.0
        nucleus_probs = []
        for token_str, p_val in sorted_probs:
            nucleus_probs.append((token_str, p_val))
            cumulative_prob += p_val
            if cumulative_prob >= self.top_p:
                break
        
        if not nucleus_probs: # Should not happen if probs_dist is not empty
            return probs_dist

        # Re-normalize
        total_prob_sum = sum(p for _, p in nucleus_probs)
        if total_prob_sum == 0:
            if not nucleus_probs: return []
            uniform_prob = 1.0 / len(nucleus_probs)
            return [(token_str, uniform_prob) for token_str, _ in nucleus_probs]
            
        return [(token_str, p / total_prob_sum) for token_str, p in nucleus_probs]

    def _apply_top_k_filter(self, probs_dist: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if self.top_k_filter is None or not probs_dist or len(probs_dist) <= self.top_k_filter:
            return probs_dist

        # Sort by probability in descending order and take top K
        top_k_probs = sorted(probs_dist, key=lambda item: item[1], reverse=True)[:self.top_k_filter]

        # Re-normalize
        total_prob_sum = sum(p for _, p in top_k_probs)
        if total_prob_sum == 0:
            if not top_k_probs: return []
            uniform_prob = 1.0 / len(top_k_probs)
            return [(token_str, uniform_prob) for token_str, _ in top_k_probs]
            
        return [(token_str, p / total_prob_sum) for token_str, p in top_k_probs]


    def sample_candidate_tokens(
        self,
        logprobs_from_api: List[Tuple[str, float]], # List of (token_str, logprob_val)
        num_candidates_to_select: int
    ) -> List[str]: # Returns list of selected raw token strings
        """
        Applies sampling parameters (temp, min_p, top_p, top_k_filter) to the
        logprobs and selects a number of candidate tokens.
        """
        if not logprobs_from_api:
            logger.warning("No logprobs provided to sample_candidate_tokens.")
            return []

        # 1. Apply temperature and normalize to get a probability distribution
        current_dist = self._apply_temperature(logprobs_from_api)
        if not current_dist: return []

        # 2. Apply min_p filtering
        current_dist = self._apply_min_p_filter(current_dist)
        if not current_dist: return []

        # 3. Apply top_p (nucleus) filtering
        current_dist = self._apply_top_p_filter(current_dist)
        if not current_dist: return []

        # 4. Apply top_k filtering
        current_dist = self._apply_top_k_filter(current_dist)
        if not current_dist: return []
        
        # At this point, current_dist is the final probability distribution
        # from which to select candidates. We select `num_candidates_to_select`
        # by taking the highest probability ones.
        
        # Sort by final probability (descending) to pick the top ones
        current_dist_sorted = sorted(current_dist, key=lambda item: item[1], reverse=True)
        
        selected_token_strings = [token_str for token_str, _ in current_dist_sorted[:num_candidates_to_select]]
        
        logger.debug(f"Sampler selected {len(selected_token_strings)} candidates: {selected_token_strings}")
        return selected_token_strings