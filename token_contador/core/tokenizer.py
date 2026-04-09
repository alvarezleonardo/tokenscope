from __future__ import annotations
import tiktoken
from token_contador.core.models import ModelInfo, TokenCount

try:
    from transformers import AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment,misc]
    _HF_AVAILABLE = False

_TIKTOKEN_ENCODINGS = {
    "tiktoken-cl100k": "cl100k_base",
    "tiktoken-o200k": "o200k_base",
}

_DEFAULT_CHARS_PER_TOKEN = 4.0
_tiktoken_cache: dict[str, tiktoken.Encoding] = {}
_hf_cache: dict[str, object] = {}


class TokenizerEngine:
    def count(self, text: str, model: ModelInfo) -> TokenCount:
        tok = model.tokenizer

        if tok in _TIKTOKEN_ENCODINGS:
            return self._count_tiktoken(text, tok)

        if tok.startswith("hf:"):
            return self._count_hf(text, tok[3:])

        # "estimated" or any other value
        chars_per_token = model.chars_per_token or _DEFAULT_CHARS_PER_TOKEN
        return self._count_estimated(text, chars_per_token)

    def _count_tiktoken(self, text: str, enc_name: str) -> TokenCount:
        enc_key = _TIKTOKEN_ENCODINGS[enc_name]
        if enc_key not in _tiktoken_cache:
            _tiktoken_cache[enc_key] = tiktoken.get_encoding(enc_key)
        tokens = len(_tiktoken_cache[enc_key].encode(text))
        return TokenCount(tokens=tokens, method="exact", tokenizer=enc_name)

    def _count_hf(self, text: str, repo_id: str) -> TokenCount:
        if not _HF_AVAILABLE:
            return self._count_estimated(text, _DEFAULT_CHARS_PER_TOKEN)
        try:
            if repo_id not in _hf_cache:
                _hf_cache[repo_id] = AutoTokenizer.from_pretrained(repo_id)
            tokens = len(_hf_cache[repo_id].encode(text))
            return TokenCount(tokens=tokens, method="exact", tokenizer=f"hf:{repo_id}")
        except Exception:
            return self._count_estimated(text, _DEFAULT_CHARS_PER_TOKEN)

    def _count_estimated(self, text: str, chars_per_token: float) -> TokenCount:
        tokens = max(1, round(len(text) / chars_per_token))
        return TokenCount(tokens=tokens, method="estimated", tokenizer="estimated")
