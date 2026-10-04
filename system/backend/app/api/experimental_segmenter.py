"""Opt-in experimental PyVi segmentation for M1 raw Vietnamese text.

This module never normalizes the submitted article for storage, alters a
checkpoint, or claims PyVi matches the unknown upstream dataset segmenter.
It rejects any unexpected change to visible Unicode characters (NFC form)
and any join boundary not arising from whitespace in the original article.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Optional


# Mirrors the surface equivalence used by the M1 segmentation drift audit.
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _visible_surface(s: str) -> str:
    return re.sub(r"[\s_]+", "", _nfc(s))


def _join_offsets(s: str) -> set[int]:
    offset = 0
    joins: set[int] = set()
    for char in _nfc(s):
        if char == "_":
            joins.add(offset)
        elif not char.isspace():
            offset += 1
    return joins


def _original_space_offsets(s: str) -> set[int]:
    offset = 0
    spaces: set[int] = set()
    pending = False
    for char in _nfc(s):
        if char.isspace():
            pending = offset > 0
        else:
            if pending:
                spaces.add(offset)
                pending = False
            offset += 1
    return spaces


def segment_raw_pyvi(text: str, tokenizer: Optional[Callable[[str], str]] = None) -> str:
    """Return experimental segmented text or raise ValueError for unsafe output.

    tokenizer is injectable for unit tests; production uses ViTokenizer.tokenize.
    Spaces and punctuation spacing can differ, but Unicode visible characters
    must match exactly after NFC. Underscores are reserved for segmentation.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("article_text must not be blank")
    if "_" in text:
        raise ValueError("Experimental raw-text mode does not accept underscores; use presegmented input instead")
    if tokenizer is None:
        from pyvi import ViTokenizer  # Lazy: no import required for baseline API tests.
        tokenizer = ViTokenizer.tokenize
    try:
        result = tokenizer(text)
    except Exception:
        raise ValueError("Experimental segmentation failed; M1 was not executed") from None
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Experimental segmenter returned no usable text")
    if _visible_surface(text) != _visible_surface(result):
        raise ValueError("Experimental segmenter changed visible article characters; M1 was not executed")
    unexpected = _join_offsets(result) - _original_space_offsets(text)
    if unexpected:
        raise ValueError("Experimental segmenter created a join outside an original word boundary; M1 was not executed")
    return result
