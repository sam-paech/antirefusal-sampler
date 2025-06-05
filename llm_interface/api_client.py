import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.models import APILogProbsResponse, APIGreedySampleResponse # Adjusted model names

logger = logging.getLogger(__name__)

_SHARED_SESSIONS: Dict[str, requests.Session] = {}

class ApiClient:
    """
    OpenAI-compatible completions client.
    Manages HTTP sessions and API requests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: int,
        pool_size: int = 20, # Default pool size
    ):
        if not base_url.endswith("/v1"):
            self.base_url = base_url.rstrip("/") + "/v1"
        else:
            self.base_url = base_url

        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.completion_endpoint = f"{self.base_url}/completions"

        session_key = f"{self.base_url}_{pool_size}"
        if session_key not in _SHARED_SESSIONS:
            _SHARED_SESSIONS[session_key] = self._build_session(pool_size)
        self._session = _SHARED_SESSIONS[session_key]

        logger.info(
            f"ApiClient initialized for model '{model_name}' at '{self.completion_endpoint}', pool_size={pool_size}"
        )

    @staticmethod
    def _build_session(pool_size: int) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5, # Shorter backoff for faster retries if appropriate
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"POST"}, # Changed from frozenset to set for broader compatibility
        )
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=retries
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _make_request(self, payload: Dict) -> Dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.lower() != "empty": # Allow "empty" or "" for no key
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Ensure common parameters are set
        payload["model"] = self.model_name
        payload.setdefault("stream", False) # Default to non-streaming

        logger.debug(f"API Request to {self.completion_endpoint}: Payload: {json.dumps(payload)[:500]}")
        
        try:
            response = self._session.post(
                self.completion_endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out after {self.timeout_seconds}s.")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"API HTTP error: {e}. Response: {e.response.text if e.response else 'No response text'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode API JSON response: {e}. Response text: {response.text if 'response' in locals() else 'No response captured'}")
            raise


    def get_next_token_logprobs(
        self,
        prompt_prefix: str,
        top_n_logprobs: int,
        sampling_params: Dict # e.g., temperature, min_p from config.sampling_beam_starters
    ) -> APILogProbsResponse:
        """
        Gets logprobs for the single next token.
        """
        payload = {
            "prompt": prompt_prefix,
            "max_tokens": 1,
            "logprobs": top_n_logprobs, # vLLM and OpenAI use 'logprobs' for count of top logprobs
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p"), # Can be None
            "top_k": sampling_params.get("top_k_filter"), # Can be None
            "min_p": sampling_params.get("min_p"), # Custom for vLLM-like, might not be std OpenAI
        }
        # Filter out None values for top_p, top_k, min_p as some APIs don't like nulls
        payload = {k: v for k, v in payload.items() if v is not None}


        api_response_data = self._make_request(payload)
        logger.debug(f"Logprobs API Raw Response: {json.dumps(api_response_data)[:1000]}")

        if not api_response_data.get("choices"):
            logger.warning("Logprobs API response contained no choices.")
            return APILogProbsResponse(logprobs=[])

        choice = api_response_data["choices"][0]
        
        logprobs_data = choice.get("logprobs")
        if not logprobs_data or not logprobs_data.get("top_logprobs"):
            logger.warning("No top_logprobs found in API response for logprobs request.")
            return APILogProbsResponse(logprobs=[])

        # top_logprobs is usually a list of dictionaries, one per generated token.
        # Since max_tokens=1, we expect a list with one element.
        # Each element is a dict like {"token_string": probability, ...}
        # Or for OpenAI: "top_logprobs": [[{"token": string, "logprob": float, "bytes": list_of_int_or_null}...]]
        
        # Handle vLLM-like structure: logprobs: { tokens: ["..."], top_logprobs: [ {token: logprob, ...}, ... ] }
        # Handle OpenAI structure: logprobs: { tokens: ["..."], top_logprobs: [ [ {token: str, logprob: float}, ... ] ]}
        
        extracted_logprobs: List[Tuple[str, float]] = []

        if "top_logprobs" in logprobs_data:
            # This could be a list of dicts (vLLM) or list of lists of dicts (OpenAI)
            first_token_logprobs_set = logprobs_data["top_logprobs"]
            
            if isinstance(first_token_logprobs_set, list) and len(first_token_logprobs_set) > 0:
                # OpenAI: list of lists. We care about the first (and only) token's alternatives.
                if isinstance(first_token_logprobs_set[0], list): 
                    alternatives_list = first_token_logprobs_set[0]
                    for alt in alternatives_list:
                        if isinstance(alt, dict) and "token" in alt and "logprob" in alt:
                            extracted_logprobs.append((alt["token"], alt["logprob"]))
                # vLLM: list of dicts, where each dict is {token_str: logprob_val, ...}
                elif isinstance(first_token_logprobs_set[0], dict):
                    alternatives_dict = first_token_logprobs_set[0]
                    for token_str, logprob_val in alternatives_dict.items():
                        extracted_logprobs.append((token_str, logprob_val))
                else:
                    logger.warning(f"Unexpected format for top_logprobs content: {first_token_logprobs_set[0]}")
            elif isinstance(first_token_logprobs_set, dict): # Direct dict for the first token (less common)
                 for token_str, logprob_val in first_token_logprobs_set.items():
                        extracted_logprobs.append((token_str, logprob_val))
            else:
                logger.warning(f"Unexpected type for top_logprobs: {type(first_token_logprobs_set)}")
        
        if not extracted_logprobs:
            logger.warning("Could not extract any logprobs from the API response.")

        # The actual token chosen by the model (if max_tokens=1)
        chosen_token_string = None
        if logprobs_data.get("tokens") and isinstance(logprobs_data["tokens"], list) and len(logprobs_data["tokens"]) > 0:
            chosen_token_string = logprobs_data["tokens"][0]
        elif choice.get("text"): # Fallback if tokens array is missing but text is present
            chosen_token_string = choice["text"]


        return APILogProbsResponse(logprobs=extracted_logprobs, chosen_token_string=chosen_token_string)


    def greedy_sample(
        self,
        prompt_prefix: str,
        num_tokens: int,
        stop_sequences: Optional[List[str]] = None
    ) -> APIGreedySampleResponse:
        """
        Greedily samples a sequence of tokens.
        """
        payload = {
            "prompt": prompt_prefix,
            "max_tokens": num_tokens,
            "temperature": 0.001, # Effectively greedy
            "top_p": 1.0,
            "logprobs": 1, # Not typically needed for greedy sampling unless debugging
        }
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        api_response_data = self._make_request(payload)
        logger.debug(f"Greedy Sample API Raw Response: {json.dumps(api_response_data)[:500]}")

        if not api_response_data.get("choices"):
            logger.warning("Greedy sample API response contained no choices.")
            return APIGreedySampleResponse(token_strings_raw=[])

        choice = api_response_data["choices"][0]
        finish_reason = choice.get("finish_reason")

        # Extract token strings. This depends on whether 'logprobs' was requested and how the API returns tokens.
        # If 'logprobs' was not requested (or is None), 'text' field is the primary source.
        # However, for consistency, it's better if the API can return token strings even for greedy.
        # If the API returns individual tokens in `choice.logprobs.tokens` even with temp=0, use that.
        # Otherwise, we have to rely on `choice.text` and acknowledge it's not "raw tokens".
        # For this sampler, we need raw tokens if possible.
        # Let's assume for now that even with temp=0, if `logprobs` (even as null) is sent,
        # some APIs might populate `choice.logprobs.tokens`. If not, this needs adjustment.

        token_strings_raw: List[str] = []
        if choice.get("logprobs") and choice["logprobs"].get("tokens"):
            token_strings_raw = choice["logprobs"]["tokens"]
        elif choice.get("text"):
            # This is a fallback and not ideal as it's not "raw" tokens.
            # It means the greedy sampling part might not align perfectly with token-by-token exploration.
            # For now, we'll use it but log a warning.
            logger.warning("Greedy sampling did not return raw token strings via logprobs; using 'text' field. This might affect precise token path reconstruction.")
            # We can't easily split 'text' into raw tokens without a tokenizer here.
            # This is a limitation if the API doesn't provide tokens for greedy sampling.
            # For now, let's assume the `BeamExplorer` will handle this by knowing the greedy part is one "chunk".
            # Or, we can make a simplifying assumption that the text is one "token" for the purpose of the path.
            # A better approach: if the API doesn't give tokens, this method should perhaps not be used,
            # or the explorer should do token-by-token greedy itself.
            # For now, let's return the text as a single "token" in the list.
            token_strings_raw = [choice["text"]] 
        
        if not token_strings_raw and choice.get("text"): # If still no tokens but text exists
             token_strings_raw = [choice.get("text","")]


        return APIGreedySampleResponse(token_strings_raw=token_strings_raw, finish_reason=finish_reason)