# utils/chat_template_helper.py -> llm_interface/chat_formatter.py
"""
Turns a Hugging-Face chat template into a plain-text prompt compatible
with the **/v1/completions** endpoint.

If *system_prompt* is supplied it is injected as a system role message
exactly once, before the user message.
"""
import threading
import logging
from typing import Tuple, Optional
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class ChatTemplateFormatter:
    _cache = {}
    _lock  = threading.Lock()

    def __init__(self,
                 model_id: str,
                 system_prompt: Optional[str] = "") -> None:
        self.model_id      = model_id
        self.system_prompt = system_prompt or "" # Ensure it's a string

        with self._lock:
            tok = self._cache.get(model_id)
            if tok is None:
                try:
                    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    self._cache[model_id] = tok
                    logger.info(f"Tokenizer for '{model_id}' loaded and cached for ChatTemplateFormatter.")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer for '{model_id}': {e}")
                    # Allow initialization to proceed, but build_prompt will fail if tokenizer is None
                    tok = None # Explicitly set to None on failure
            else:
                logger.debug(f"Using cached tokenizer for '{model_id}' in ChatTemplateFormatter.")
        self.tokenizer = tok

        if self.tokenizer:
            self._prefix, self._middle, self._sys_placeholder = self._extract_segments()
        else:
            # Set defaults if tokenizer failed to load, to prevent crashes later
            self._prefix, self._middle, self._sys_placeholder = "", "", ""
            logger.warning(f"ChatTemplateFormatter for '{model_id}' initialized without a valid tokenizer. Formatting will be basic.")


    # ------------------------------------------------------------- #
    # internal helpers                                              #
    # ------------------------------------------------------------- #
    def _extract_segments(self) -> Tuple[str, str, str]:
        if not self.tokenizer:
            return "", "", "" # Should not happen if constructor checks

        ph_user = "__PLACEHOLDER_USER__"
        ph_asst = "__PLACEHOLDER_ASST__" # This will mark where the assistant's generation starts
        ph_sys  = "__PLACEHOLDER_SYS__"

        messages = []
        if self.system_prompt: # Only add system message if system_prompt is non-empty
            messages.append({"role": "system",    "content": ph_sys})
        messages.extend([
            {"role": "user",      "content": ph_user},
            # We need to ensure the template adds the prompt for assistant generation
        ])

        # add_generation_prompt=True is crucial for /completions endpoint
        # as it appends the tokens that signal the start of the assistant's reply.
        try:
            tpl_with_gen_prompt = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": ""}], # Empty assistant content to get prompt
                tokenize=False,
                add_generation_prompt=True # This should add the assistant's turn start
            )
            
            # Simpler extraction: find user placeholder, the rest is prefix and middle
            # The "middle" part is what comes after the user prompt and before the assistant starts generating.
            # This is often just the assistant role tokens.

            # Create a version of the template with only user message to find its end
            user_only_messages = []
            if self.system_prompt:
                 user_only_messages.append({"role": "system", "content": ph_sys})
            user_only_messages.append({"role": "user", "content": ph_user})
            
            templated_user_part = self.tokenizer.apply_chat_template(
                user_only_messages,
                tokenize=False,
                add_generation_prompt=False # Don't add assistant prompt here
            )

            # The full template with generation prompt for an empty assistant message
            full_template_for_empty_assistant = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": ""}], # empty assistant message
                tokenize=False,
                add_generation_prompt=True # This is key
            )
            
            # Prefix is everything before the user's actual content
            # Middle is everything after user's content and before where assistant generation would start
            
            # Find where ph_user starts in templated_user_part
            idx_user_start_in_user_part = templated_user_part.find(ph_user)
            if idx_user_start_in_user_part == -1:
                logger.error("User placeholder not found in user-only template part.")
                raise ValueError("User placeholder not found in template (user-only part).")

            prefix = templated_user_part[:idx_user_start_in_user_part]
            
            # To find 'middle', we look at the full template that prompts for an assistant response.
            # 'middle' is the part between the end of the user's content and the start of the assistant's actual generation.
            # In `full_template_for_empty_assistant`, replace ph_user with empty string to isolate prefix and middle.
            
            # Find where ph_user ends in the full template
            idx_user_start_in_full = full_template_for_empty_assistant.find(ph_user)
            if idx_user_start_in_full == -1:
                logger.error("User placeholder not found in full template part.")
                raise ValueError("User placeholder not found in template (full part).")
            
            idx_user_end_in_full = idx_user_start_in_full + len(ph_user)
            
            middle = full_template_for_empty_assistant[idx_user_end_in_full:]

            logger.debug(f"Chat template segments for '{self.model_id}': prefix='{prefix[:50]}...', middle='{middle[:50]}...'")
            return prefix, middle, ph_sys

        except Exception as e:
            logger.error(f"Error extracting chat template segments for '{self.model_id}': {e}", exc_info=True)
            # Fallback to basic formatting if template application fails
            if self.system_prompt:
                return f"{ph_sys}\nUser: ", "\nAssistant: ", ph_sys # Basic fallback
            return "User: ", "\nAssistant: ", ph_sys


    # ------------------------------------------------------------- #
    # public API                                                    #
    # ------------------------------------------------------------- #
    def build_prompt(self,
                     user_prompt: str,
                     assistant_so_far: str = "") -> str:
        """
        Returns a string formatted for the /completions endpoint.
        Example: "System: ...\nUser: How are you?\nAssistant: I am doing well."
        If assistant_so_far is empty, it should end with the prompt for the assistant to start.
        """
        if not self.tokenizer:
            # Basic fallback if tokenizer failed to load
            logger.warning(f"Using basic formatting for '{self.model_id}' due to tokenizer load failure.")
            formatted_prompt = ""
            if self.system_prompt:
                formatted_prompt += f"{self.system_prompt}\n"
            formatted_prompt += f"User: {user_prompt}\nAssistant:"
            if assistant_so_far:
                formatted_prompt += f" {assistant_so_far}"
            return formatted_prompt

        prefix_processed = self._prefix
        if self.system_prompt and self._sys_placeholder in prefix_processed:
            prefix_processed = prefix_processed.replace(self._sys_placeholder, self.system_prompt)
        elif self.system_prompt and self._sys_placeholder not in prefix_processed:
            # If system prompt is provided but placeholder wasn't found (e.g. due to template structure)
            # Prepend it simply. This might not be ideal for all templates.
            prefix_processed = f"{self.system_prompt}\n{prefix_processed}"


        # If assistant_so_far is empty, `middle` should contain the assistant prompt.
        # If assistant_so_far is not empty, we just append it.
        if assistant_so_far:
            return f"{prefix_processed}{user_prompt}{self._middle}{assistant_so_far}"
        else:
            # self._middle should already contain the part that prompts the assistant to speak
            return f"{prefix_processed}{user_prompt}{self._middle}"