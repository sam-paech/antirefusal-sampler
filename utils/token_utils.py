import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Mojibake fixing helpers from antislop-vllm/state/generation_state.py       #
# --------------------------------------------------------------------------- #
def _build_u2b_full():
    # GPT/OpenAI table
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    u2b = {ord(chr(c)): b for b, c in zip(bs, cs)}
    # Qwen/Mistral table (bytes 0‑255 → U+2500+byte)
    for b in range(256):
        u2b[0x2500 + b] = b
    return u2b

_U2B = _build_u2b_full()

def fix_mojibake(text: str) -> str:
    buf, changed = bytearray(), False
    for ch in text:
        cp = ord(ch)
        if cp in _U2B:
            buf.append(_U2B[cp]); changed = True
        else:
            buf.extend(ch.encode('utf-8'))
    if not changed:
        return text
    try:
        return buf.decode('utf-8')
    except UnicodeDecodeError:
        # logger.warning(f"UnicodeDecodeError during mojibake fix for text: '{text[:50]}...'")
        return text # Return original if decode fails

def fix_mojibake_iter(text: str, max_rounds: int = 3) -> str:
    prev = text
    for _ in range(max_rounds):
        cur = fix_mojibake(prev)
        if cur == prev:
            return cur
        prev = cur
    return prev

def decode_token(token_str: str) -> str:
    """
    Decodes a raw token string from an LLM API.
    Handles common space/newline markers and attempts to fix mojibake.
    """
    if not token_str:
        return ""

    # Attempt to fix mojibake first
    # This order might be important depending on how markers interact with mojibake
    repaired_token = fix_mojibake_iter(token_str)

    # Common replacements for special characters representing spaces/newlines
    # Order of replacement can matter.
    # Specific markers depend on the tokenizer used by the model.
    # Add more replacements as needed based on observed tokens.
    # Example markers:
    #   'Ġ' (Llama, Mistral, etc. for leading space)
    #   'Ċ' (Newline)
    #   ' ' (Underscore for space in SentencePiece, e.g., XLNet, T5)
    #   '\u2581' (Lower one eighth block, sometimes used for space)

    decoded = repaired_token
    decoded = decoded.replace("Ġ", " ")
    decoded = decoded.replace("Ċ", "\n")
    # SentencePiece space marker (often at the beginning of words)
    # If ' ' is used as a space marker, it might conflict if actual underscores are part of tokens.
    # This needs to be handled carefully based on the specific tokenizer.
    # For now, let's assume ' ' is a space marker if it appears.
    # decoded = decoded.replace(" ", " ") # This is redundant if 'Ġ' is already handled

    # Some models might use other unicode characters for spaces, e.g. \u2581
    # decoded = decoded.replace("\u2581", " ")

    return decoded

def decode_token_path(token_path_raw: list[str]) -> str:
    """Decodes a list of raw token strings into a single text string."""
    return "".join([decode_token(t) for t in token_path_raw])