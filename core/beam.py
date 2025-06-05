from dataclasses import dataclass, field
from typing import List

@dataclass
class Beam:
    tokens_raw: List[str]           # full path, token-strings straight from API
    decoded_words: List[str]        # cleaned words (for n-gram bookkeeping)
    depth: int                      # 0-based split depth
    is_refusal: bool | None = None  # filled once classified
    conf: float | None = None
