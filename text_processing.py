import hashlib
import json
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


NEGATION_WORDS = frozenset({"not", "no", "never", "without"})
CUSTOM_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - NEGATION_WORDS)
CUSTOM_STOP_WORDS_LIST = sorted(CUSTOM_STOP_WORDS)


def clean_tokens(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [word for word in text.split() if word not in CUSTOM_STOP_WORDS]


def clean_token_group_id(text: str) -> str:
    """Return a stable identifier for a message's cleaned-token sequence."""
    token_sequence = json.dumps(
        clean_tokens(text),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(token_sequence.encode("utf-8")).hexdigest()
    return f"clean_tokens_sha256:{digest}"
