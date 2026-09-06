# Experiment 05 — smoke sweep. **This is not a result.**

These files are the output of `uv run python experiments/05_lm_training/run.py --sweep smoke`,
run on an M-series laptop's MPS device on 2026-09-06 at commit `b13f6e0`. Its only purpose
is to prove that the whole pipeline — corpus encoding, training, periodic bits-per-character
evaluation, checkpointing, aggregation and both figures — runs end to end **before a GPU is
rented**.

It is not a measurement of anything about tokenizers, for three reasons stated in
`config.yaml` itself:

- the model is two layers wide 128 (8.6 M parameters), far too small;
- the budget is 300 steps, far too short;
- the sweep is budgeted in **steps**, so the three arms do not even see equal bytes — a
  denser vocabulary sees fewer characters in the same 300 steps, which is exactly the
  comparison the real sweep is designed to avoid.

One seed, not the three CLAUDE.md §2.6 requires of a real run.

For the record, the in-domain (`heldout_dcs`) BPC it produced was `T1_bpe_raw_64k_dcs`
2.826, `T5_morphbpe_rawseg_64k_dcs` 2.860, `T7_byt5` 3.793. Do not cite these numbers.

The real sweep — 51 runs, two tracks, three seeds, 50 M and 125 M parameters — is described
in `experiments/05_lm_training/README.md`, along with the rented-GPU procedure and a
budget of roughly 18–26 A100-hours.

Files: `results.json` (aggregated across the three runs), `config.yaml` (the copy of
`smoke.yaml` the run used), `sweep_plan.json` (what it planned to run), and the two figures
`bpc_curves` / `bpc_final` in PDF and PNG.
