# Decision log

Append-only. Newest entries at the bottom. Format per CLAUDE.md §11.

## 2026-09-03 — Use SLP1 as internal encoding
Why: CharSS 2024 and ByT5-Sanskrit both report SLP1 outperforms Devanagari for neural models; it is lossless and ASCII.
Alternatives: IAST (diacritics complicate tokenizer training), Devanagari (byte-level fragmentation).
Reversible: yes, at cost of re-running tokenizer training.

## 2026-09-03 — Manage environment with uv installed via `python3 -m pip install --user uv`
Why: `uv` was not present on the machine. `pip install --user` bootstraps the tool itself into `~/.local/bin` (already on PATH) without touching the project environment, which stays fully `uv`-managed from `pyproject.toml` + `uv.lock`. Installed version: uv 0.12.9.
Alternatives: the astral install script `curl -LsSf https://astral.sh/uv/install.sh | sh` (equivalent result, extra network trust); Homebrew (adds a package manager dependency).
Reversible: yes, `pip uninstall uv` and reinstall by any other method; no project files depend on how uv was obtained.

## 2026-09-03 — Add `.DS_Store` to `.gitignore`
Why: macOS Finder writes `.DS_Store` into every directory it opens; several were picked up by `git add -A` during the initial scaffold. They carry no project information and would churn every commit.
Alternatives: a global `~/.gitignore_global` (does not travel with the repo, so other contributors would still commit them).
Reversible: yes, delete the line.

## 2026-09-03 — SLP1 roundtrip is guaranteed for Devanagari/IAST only, not for embedded Latin text; no NFC normalisation
Why: SLP1 is itself an ASCII scheme, so `from_slp1` must read Latin letters as phonemes, ASCII digits as Devanagari digits, and `'` `.` `~` `|` as avagraha, danda, candrabindu and Vedic ḻh. A line mixing English with Devanagari therefore cannot roundtrip — `from_slp1` has no way to tell "The" from a run of SLP1 phonemes — and fixture line 10 of `tests/fixtures/devanagari_sample.txt` is kept as a strict `xfail` documenting that boundary. Unicode NFC normalisation was *not* added to `to_slp1`: the eleven pure-Devanagari fixture lines roundtrip without it, and normalising inside `to_slp1` would in fact break identity roundtrip for any decomposed input (the output would come back composed). Callers that need decomposed text handled must normalise on ingest, in `data/`.
Alternatives: wrapping Latin runs in `sanscript` toggle markers (`##`) so they survive the reverse trip — rejected, it injects non-Sanskrit markers into the SLP1 string that tokenizers are trained on, which is the whole point of the encoding. Per CLAUDE.md §2.3 the original script is stored alongside the SLP1 form instead.
Reversible: yes; adding NFC or a toggle-based passthrough is a local change to `encoding.py` plus its tests.

## 2026-09-03 — Load FLORES-200 from the official NLLB tarball, not from a Hugging Face dataset id
Why: `load_flores` tries four sources in order and the first three all failed on this machine, so the download that produced `data/raw/flores/devtest.jsonl` (1012 devtest sentences, `san_Deva`/`hin_Deva`/`eng_Latn`) came from source 4, `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (sha256 `b8b0b767...944011f6`). Failures, in order: (1) `openlanguagedata/flores_plus` — `DatasetNotFoundError: Dataset 'openlanguagedata/flores_plus' is a gated dataset on the Hub. You must be authenticated to access it.` (no `HF_TOKEN` here); (2) `facebook/flores` — same `DatasetNotFoundError`, now also gated, and independently unusable because `datasets` 5.0.1 has dropped `trust_remote_code` and loading scripts ("trust_remote_code is not supported anymore"); (3) `Muennighoff/flores200` — `RuntimeError: Dataset scripts are no longer supported, but found flores200.py`, the same `datasets` 5.x restriction. The tarball is the upstream release the Hub copies are all derived from, needs no Hub account, and is read with `urllib.request` + `tarfile` from the standard library, so it adds no dependency. It is extracted through `tarfile.extractfile` for only the requested `<split>/<lang>.<split>` members, so no archive path is ever written to disk.
Alternatives: obtaining an `HF_TOKEN` and accepting the `flores_plus` licence (better long-term — `flores_plus` is the maintained release with per-sentence metadata — but it needs a credential this machine does not have, and gating makes the pipeline non-reproducible for anyone without one); pinning `datasets<4` so loading scripts still run (rejected, it would drag the whole environment backwards for one corpus).
Reversible: yes. The four sources are tried in order on every call, so simply setting `HF_TOKEN` and re-running `load_flores` switches the download to `flores_plus` with no code change; the sentences are the same FLORES-200 devtest either way, and only `data/README.md` would need its provenance row updated.
