"""Sandhi-splitting wrappers (ByT5-Sanskrit, TransLIST).

`SandhiSplitter` is the project's one sandhi-reversal interface: Devanagari in, SLP1 with
a space at every segment boundary out, every result cached in a `SplitCache`. The model,
its IAST task format and its 512-byte window stay inside `byt5.py` (docs/decisions.md,
"Experiment 03 sandhi splitter"); a TransLIST implementation, if ablation A3 ever needs
one, goes beside it behind the same two methods.

`reconcile` is the other half of the story: the model's output is not a re-segmentation of
its input (it drops punctuation and the occasional loanword), so the text Experiment 03
measures is that output aligned back onto the raw sentence — see `reconcile.py`.
"""

from sanskrit_tok.sandhi.byt5 import SandhiSplitter
from sanskrit_tok.sandhi.cache import SplitCache
from sanskrit_tok.sandhi.reconcile import ReconcileResult, reconcile

__all__ = ["ReconcileResult", "SandhiSplitter", "SplitCache", "reconcile"]
