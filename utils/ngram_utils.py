import logging
import string
from typing import List, Set, Tuple, Dict

import nltk

logger = logging.getLogger(__name__)

_NLTK_DATA_ENSURED = False
_NLTK_DOWNLOAD_LOCK = nltk.downloader.Lock() # Use NLTK's lock for thread-safety

def _ensure_nltk_data():
    global _NLTK_DATA_ENSURED
    if _NLTK_DATA_ENSURED:
        return

    with _NLTK_DOWNLOAD_LOCK:
        if _NLTK_DATA_ENSURED: # Double check after acquiring lock
            return

        data_to_check_download = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords'
        }
        all_data_present = True
        for name, path_check in data_to_check_download.items():
            try:
                nltk.data.find(path_check)
            except (nltk.downloader.DownloadError, LookupError):
                logger.info(f"NLTK '{name}' resource not found. Attempting download...")
                try:
                    nltk.download(name, quiet=True)
                    logger.info(f"NLTK '{name}' downloaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to download NLTK '{name}': {e}. NgramUtil might not work correctly.")
                    all_data_present = False
        
        if all_data_present:
            _NLTK_DATA_ENSURED = True
        else:
            logger.warning("Not all NLTK data could be ensured. Ngram functionality may be impaired.")


class NgramUtil:
    def __init__(self, language: str = "english"):
        _ensure_nltk_data()
        self.language = language
        try:
            self.stopwords_set = set(nltk.corpus.stopwords.words(language))
        except Exception as e:
            logger.warning(f"Could not load stopwords for language '{language}': {e}. Stopword removal will be disabled for this instance.")
            self.stopwords_set = set()
        self.punctuation_set = set(string.punctuation)

    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenizes text, converts to lowercase, and removes punctuation and stopwords."""
        if not text:
            return []
        try:
            tokens = nltk.word_tokenize(text.lower())
            # Filter out punctuation and stopwords
            cleaned_tokens = [
                token
                for token in tokens
                if token not in self.punctuation_set and token not in self.stopwords_set and token.isalnum() # Ensure alphanumeric
            ]
            return cleaned_tokens
        except Exception as e:
            logger.error(f"Error during tokenization/cleaning: {e}", exc_info=True)
            return []


    def generate_ngrams(self, text: str, n_values: List[int]) -> Dict[int, List[Tuple[str, ...]]]:
        """
        Generates n-grams of specified sizes from the text after cleaning.
        Returns a dictionary where keys are n-gram sizes and values are lists of n-gram tuples.
        """
        cleaned_tokens = self._tokenize_and_clean(text)
        ngrams_dict = {}
        for n in n_values:
            if n <= 0:
                continue
            if len(cleaned_tokens) < n:
                ngrams_dict[n] = []
                continue
            try:
                n_grams = list(nltk.ngrams(cleaned_tokens, n))
                ngrams_dict[n] = n_grams
            except Exception as e:
                logger.error(f"Error generating {n}-grams: {e}", exc_info=True)
                ngrams_dict[n] = []
        return ngrams_dict

    def get_relevant_ngrams_for_token(self, previous_tokens_decoded: List[str], current_token_decoded: str, n_values: List[int]) -> Dict[str, List[str]]:
        """
        Generates n-grams that are formed by adding the current_token_decoded
        to the end of the previous_tokens_decoded sequence.
        Returns a dict like {"bigram": ["prev_tok_word_current_tok_word"], "trigram": [...]}.
        N-grams are returned as space-separated strings.
        """
        if not current_token_decoded.strip(): # Ignore if current token is just whitespace
            return {f"{n}gram": [] for n in n_values}

        # Tokenize the current token itself, as it might be multiple words after decoding
        current_token_words = self._tokenize_and_clean(current_token_decoded)
        if not current_token_words: # If current token becomes empty after cleaning
            return {f"{n}gram": [] for n in n_values}

        # Combine previous tokens (already cleaned words) with the new words from current_token
        # We assume previous_tokens_decoded are individual words, already cleaned.
        # If previous_tokens_decoded are full phrases, they need to be tokenized and cleaned first.
        # For simplicity, let's assume previous_tokens_decoded are cleaned words.
        
        # To form n-grams ending with current_token_words, we need context from previous_tokens_decoded.
        # The `previous_tokens_decoded` here should be a list of *words* from the decoded path, not raw tokens.
        # This part needs careful handling of how `previous_tokens_decoded` is constructed.
        # For now, let's assume `previous_tokens_decoded` is a list of cleaned words from the path *before* the current split token.

        full_word_sequence = previous_tokens_decoded + current_token_words
        
        output_ngrams = {}

        for n in n_values:
            if n <= 0:
                continue
            output_ngrams[f"{n}gram"] = []
            if len(full_word_sequence) >= n:
                # We are interested in n-grams that *end* with the last word of current_token_words
                # and include (n-1) words before it from the full_word_sequence.
                # Consider all n-grams ending at each word of current_token_words
                for i in range(len(current_token_words)):
                    # The index in full_word_sequence corresponding to the end of the i-th word of current_token_words
                    end_idx_in_full_sequence = len(previous_tokens_decoded) + i
                    if end_idx_in_full_sequence + 1 >= n:
                        ngram_tuple = tuple(full_word_sequence[end_idx_in_full_sequence - n + 1 : end_idx_in_full_sequence + 1])
                        output_ngrams[f"{n}gram"].append(" ".join(ngram_tuple))
        
        # Deduplicate
        for key in output_ngrams:
            output_ngrams[key] = sorted(list(set(output_ngrams[key])))
            
        return output_ngrams