"""nanoGPT-style language-model training and bits-per-character evaluation.

Experiment 05's half of the thesis: Experiment 04 measured what the tokenizers *do* to
Sanskrit text; this package measures whether it matters to a model that has to predict it.
The headline is bits per SLP1 character (`bpc`), never perplexity — different vocabularies
make perplexity incomparable (CLAUDE.md §2.2), and BPC removes the vocabulary from the
denominator entirely.

Four modules and one vendored file:

- `model.py` — nanoGPT's `GPT`, vendored byte-for-byte under its MIT licence; its header
  records the upstream commit and this file's sha256.
- `config.py` — `TrainConfig`, the three model sizes, and the device/dtype resolution that
  lets the same config run on this laptop's MPS and on a rented A100.
- `data.py` — a corpus and a tokenizer arm in, a memmapped `.bin` of token ids plus a
  `.meta.json` out; one end-of-line token per line, dtype chosen by vocabulary size.
- `bpc.py` — the pure arithmetic: summed nats over SLP1 characters, in bits.
- `train.py` — the loop, the evaluator that produces those nats, and the `results.json`,
  `config.yaml`, `curve.jsonl` and `ckpt.pt` every run leaves behind.

Nothing is imported eagerly: `torch` is heavy and most of this repository's tests never
touch it, so import the submodule you need.
"""
