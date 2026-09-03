# sanskrit-tok

This project tests whether Sanskrit's morphological density survives subword tokenization,
measured per unit of meaning rather than per word, and whether reversing sandhi and
constraining BPE merges to gold morpheme boundaries recovers the lost efficiency.
Install with `uv sync`, and run the test suite with `uv run pytest`.
Reproduce Experiment 01 with `uv run python experiments/01_baseline_penalty/run.py`.
