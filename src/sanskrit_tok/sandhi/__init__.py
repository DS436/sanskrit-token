"""Sandhi-splitting wrappers (ByT5-Sanskrit, TransLIST).

`SandhiSplitter` is the project's one sandhi-reversal interface: Devanagari in, SLP1 with
a space at every segment boundary out, every result cached in a `SplitCache`. The model,
its IAST task format and its 512-byte window stay inside `byt5.py` (docs/decisions.md,
"Experiment 03 sandhi splitter"); a TransLIST implementation, if ablation A3 ever needs
one, goes beside it behind the same two methods.
"""

from sanskrit_tok.sandhi.byt5 import SandhiSplitter
from sanskrit_tok.sandhi.cache import SplitCache

__all__ = ["SandhiSplitter", "SplitCache"]
