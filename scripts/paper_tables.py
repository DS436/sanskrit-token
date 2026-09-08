"""Generate every LaTeX table and every prose number of `paper/1a` from the snapshot.

The manuscript in `paper/1a/` contains no typed-in numbers. Each table is written here
from `results/01_baseline_penalty/results.json` and `results/02_tpp_parallel/results.json`
(the tracked snapshot, see `results/README.md`), and every figure in the prose is a macro
defined in the generated `tables/numbers.tex`.

Framing rules obeyed here, from CLAUDE.md §2:

* Fertility is always reported and never headlined. It appears in Table 2 and in the last
  column of the deployed-practice tables, alongside the ratio it would have inflated.
* Off-the-shelf arms (T0, T3) are deployed practice, never a controlled comparison. Only
  the T1/T2 over E1 pairs are matched on algorithm, vocabulary size and training corpus.
* Ratios print to three decimals, fertility and compression to two.

The length writers (`tpp_by_length.tex`, `tpp_by_length_sa.tex`,
`tpp_by_length_all.tex`) and `renyi.tex` emit a one-line LaTeX comment when the
snapshot has no `tpp_by_length` / `tpp_by_length_sa` / `renyi` key, so this script
already runs against a snapshot that lacks them. Every reader here addresses the keys
it needs by name and ignores the rest, so a snapshot that gains keys (Experiment 02 is
still adding analyses) changes nothing that is not read.

Usage::

    uv run python scripts/paper_tables.py
    uv run python scripts/paper_tables.py --results-dir results --out-dir paper/1a/tables
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("paper_tables")

#: The deployed general-purpose arms, in the order Experiment 01 reports them.
T0_ARMS = ("T0_o200k", "T0_llama4", "T0_gemma3", "T0_gpt2")
#: The deployed Indic arms available in the snapshot.
T3_ARMS = ("T3_sarvam", "T3_sutra", "T3_brahmic131k")
#: The trained Sanskrit arms. Provisional: trained on the parallel corpora, not on a
#: monolingual corpus (`experiments/02_tpp_parallel/README.md`, caveats).
TRAINED_ARMS = (
    "T1_bpe_raw_32k",
    "T1_bpe_raw_64k",
    "T1_bpe_raw_128k",
    "T2_unigram_raw_32k",
    "T2_unigram_raw_64k",
    "T2_unigram_raw_128k",
)
#: The byte-level reference: UTF-8 itself, 256 ids, no training. Not a trained arm and not
#: deployed practice, so it is named separately everywhere it appears.
BYTE_ARM = "T7_byt5"
#: Scored on both sides of the same pairs, the byte arm is the two sides' byte ratio.
BYTE_PAIR = f"{BYTE_ARM}/{BYTE_ARM}"
#: Arms whose ≥200k vocabulary makes them current practice rather than a period piece.
LARGE_VOCAB_T0 = ("T0_o200k", "T0_llama4", "T0_gemma3")

#: The matched pairs this paper reports, in reporting order: each trained Sanskrit arm
#: over its pair-matched English control (`E1_*`, the English side of the same sentences)
#: and over its byte-matched one (`E1_*_bm`, a subsample of that English side cut to the
#: Sanskrit corpus's byte count). Named explicitly rather than read off
#: `config["controlled_pairs"]` because Experiment 02 keeps adding pairs and each needs
#: its own prose before it can appear in a table. A pair the snapshot does not carry is
#: skipped; a pair the snapshot gains and this tuple does not name is ignored.
CONTROLLED_PAIRS: tuple[str, ...] = (
    "T1_bpe_raw_32k/E1_bpe_32k",
    "T1_bpe_raw_32k/E1_bpe_32k_bm",
    "T1_bpe_raw_64k/E1_bpe_64k",
    "T1_bpe_raw_64k/E1_bpe_64k_bm",
    "T1_bpe_raw_128k/E1_bpe_128k",
    "T1_bpe_raw_128k/E1_bpe_128k_bm",
    "T2_unigram_raw_32k/E1_unigram_32k",
    "T2_unigram_raw_32k/E1_unigram_32k_bm",
    "T2_unigram_raw_64k/E1_unigram_64k",
    "T2_unigram_raw_64k/E1_unigram_64k_bm",
    "T2_unigram_raw_128k/E1_unigram_128k",
    "T2_unigram_raw_128k/E1_unigram_128k_bm",
)

#: The vocabulary sizes, smallest first, as the arm names spell them.
VOCAB_TOKENS = ("32k", "64k", "128k")
#: The two vocabulary sizes at which every pair is size-matched on both sides, and at
#: which the paper's prose-corpus statement holds. Read off the pair names, not hard-coded
#: as a set of pairs, so that adding a pair to `CONTROLLED_PAIRS` needs no second edit.
SMALL_VOCAB_TOKENS = ("32k", "64k")
#: The third size, added in the 2026-09-08 wave, at which the in-domain prose verdict
#: changes for BPE.
HUGE_VOCAB_TOKEN = "128k"

#: The matched pairs the length strata carry. Experiment 02's `length_strata_skip_arms`
#: excludes the 128k arms, the byte-matched controls and the byte reference from the
#: stratification, so §5.5's tables and macros run over these four and no more.
LENGTH_PAIRS: tuple[str, ...] = (
    "T1_bpe_raw_32k/E1_bpe_32k",
    "T1_bpe_raw_64k/E1_bpe_64k",
    "T2_unigram_raw_32k/E1_unigram_32k",
    "T2_unigram_raw_64k/E1_unigram_64k",
)

#: The arms the R\'enyi appendix tabulates. It is a diagnostic that supports no claim
#: here (Cognetta et al. 2024), so it is reported for the arm set it was introduced for —
#: the size-matched 32k and 64k arms and their pair-matched controls — rather than
#: extended over the byte-matched and 128k arms of §5.3. The caption says so.
RENYI_TRAINED_ARMS = (
    "T1_bpe_raw_32k",
    "T1_bpe_raw_64k",
    "T2_unigram_raw_32k",
    "T2_unigram_raw_64k",
)
RENYI_ENGLISH_ARMS = ("E1_bpe_32k", "E1_bpe_64k", "E1_unigram_32k", "E1_unigram_64k")

#: Below this many pairs a length bin's ratio is not printed in a body table: its interval
#: spans more than an order of magnitude and the cell distracts from the rest of the row.
#: The value stays in the appendix table, and the bin's size stays in the `n` row.
LENGTH_BODY_SUPPRESS_BELOW = 10

#: Corpus display names, in the config's order (prose before verse, CLAUDE.md §2.7).
CORPUS_LABELS: dict[str, str] = {
    "samayik_test": "S\\=amayik test (prose)",
    "samayik_test_ood": "S\\=amayik test\\_ood (prose, OOD)",
    "itihasa_test": "Itih\\=asa test (verse)",
    "flores_devtest": "FLORES devtest",
}

#: Shorter forms of the same labels, for tables that would otherwise run past the margin.
SHORT_CORPUS_LABELS: dict[str, str] = {
    "samayik_test": "S\\=amayik test",
    "samayik_test_ood": "S\\=amayik test\\_ood",
    "itihasa_test": "Itih\\=asa test",
    "flores_devtest": "FLORES devtest",
}

#: Pre-registered predictions (outline §8) as numbers, so the prose can cite them from a
#: macro like everything else. These are the paper's own pre-registration, not results.
PREREG_FERTILITY_LO = 5
PREREG_FERTILITY_HI = 12
PREREG_HINDI_FERTILITY_LO = 2
PREREG_HINDI_FERTILITY_HI = 4
PREREG_PARITY_THRESHOLD = "1.5"

#: Sizes of the tokenizer-training splits, in aligned pairs, quoted from the provenance
#: table of `data/README.md` (Sāmayik commit `f87d549`, Itihāsa commit `37df077`). They are
#: taken from that file rather than measured here because the training manifests live under
#: the gitignored `data/processed/` and are not part of the tracked results snapshot, so
#: this script cannot read them. The trained Sanskrit arms saw the Sanskrit side of these
#: splits and the E1 control arms the English side, in both cases after exclusion filtering
#: against `data/exclusion_hashes.txt` and exact deduplication.
SAMAYIK_TRAIN_PAIRS = 43_493
ITIHASA_TRAIN_PAIRS = 75_161

#: Sizes of the three tokenizer-training corpora in UTF-8 bytes, and the line count of the
#: byte-matched English subsample, quoted from `experiments/02_tpp_parallel/README.md`
#: ("Byte-matched control") and from the matching `docs/decisions.md` entry of 2026-09-08.
#: They are constants for the same reason the pair counts above are: the training corpora
#: and their manifests live under the gitignored `data/processed/`, and the tracked results
#: snapshot records vocabulary sizes and hashes but not corpus sizes, so this script cannot
#: read them. `tests/test_paper_tables.py` greps the README for each of them.
SANSKRIT_TRAIN_BYTES = 11_209_356
ENGLISH_TRAIN_BYTES = 16_554_871
ENGLISH_BM_TRAIN_BYTES = 11_209_371
ENGLISH_BM_TRAIN_LINES = 78_624
ENGLISH_TRAIN_LINES = 116_127

#: The two length stratifications, as (results key, bin-edge config key, binned side).
#: Both are reported because binning on one side selects that side's noise into the bin,
#: which biases the ratio in a known direction: down for English bins, up for Sanskrit
#: bins (`docs/decisions.md`, "Length strata reported under both ... binning").
LENGTH_STRATA: tuple[tuple[str, str, str], ...] = (
    ("tpp_by_length", "length_bin_edges", "English"),
    ("tpp_by_length_sa", "length_bin_edges_sa", "Sanskrit"),
)

#: The corpora the body's length tables carry: the two the verse/prose reading compares,
#: prose first (CLAUDE.md §2.7). All four are in the appendix table.
LENGTH_BODY_CORPORA = ("samayik_test", "itihasa_test")

#: The prose corpus and the verse corpus of that comparison, in that order.
LENGTH_PROSE_CORPUS = "samayik_test"
LENGTH_VERSE_CORPUS = "itihasa_test"

#: The α at which the prose of §5.6 reads the Rényi table. Both α values are tabulated.
RENYI_PROSE_ALPHA = "2.5"
#: The Rényi corpus the prose reads: the primary prose corpus, as everywhere else.
RENYI_PROSE_CORPUS = "samayik_test"

#: Every macro `numbers.tex` defines. `tests/test_paper_tables.py` asserts that main.tex
#: uses no `\num...` macro outside this list and that every entry here is emitted.
MACROS: tuple[str, ...] = (
    "numFloresSents",
    "numSamayikTest",
    "numSamayikOod",
    "numItihasaTest",
    "numBootstrap",
    "numBootstrapSeed",
    "numCiLevel",
    "numSaWords",
    "numEnWords",
    "numHiWords",
    "numParitySaEnLo",
    "numParitySaEnHi",
    "numParitySaEnGptTwo",
    "numParitySaHiLo",
    "numParitySaHiHi",
    "numParitySaHiLargeLo",
    "numParitySaHiLargeHi",
    "numFertSaLo",
    "numFertSaHi",
    "numFertSaGptTwo",
    "numFertSaGptTwoSlpOne",
    "numFertRatioSaEnLo",
    "numFertRatioSaEnHi",
    "numFertRatioSaHiLo",
    "numFertRatioSaHiHi",
    "numCompSaOrigLo",
    "numCompSaOrigHi",
    "numCompEnLo",
    "numCompEnHi",
    "numCompSaSlpLo",
    "numCompSaSlpHi",
    "numPreregFertLo",
    "numPreregFertHi",
    "numPreregHindiFertLo",
    "numPreregHindiFertHi",
    "numPreregParityThreshold",
    "numDeployedLo",
    "numDeployedHi",
    "numDeployedSamayikOTwoHundredK",
    "numDeployedSamayikOTwoHundredKCi",
    "numTppBpeSixtyFourDeployed",
    "numTppBpeSixtyFourDeployedCi",
    "numTppControlledBest",
    "numTppControlledBestCi",
    "numTppControlledBpeThirtyTwo",
    "numTppControlledSamayikLo",
    "numTppControlledSamayikHi",
    "numTppControlledOodLo",
    "numTppControlledOodHi",
    "numTppControlledFloresLo",
    "numTppControlledFloresHi",
    "numTppControlledItihasaLo",
    "numTppControlledItihasaHi",
    "numTppControlledItihasaAllLo",
    "numTppControlledItihasaAllHi",
    "numTppControlledHugeOodLo",
    "numTppControlledHugeOodHi",
    "numControlledPairs",
    "numControlledPairsAll",
    "numControlledPairsSmall",
    "numControlledPairsHuge",
    # The byte-matched control.
    "numBmLines",
    "numBmLinePct",
    "numBytesSanskritTrain",
    "numBytesEnglishTrain",
    "numBytesEnglishBmTrain",
    "numBytesEnglishExcessPct",
    "numBmMoveCount",
    "numBmMaxMove",
    "numBmMedianMove",
    "numBmDownwardCount",
    "numBmVerdictChanges",
    # The vocabulary sweep.
    "numVocabHuge",
    "numTppBpeHugeSamayik",
    "numTppBpeHugeSamayikCi",
    "numTppBpeHugeSamayikBm",
    "numTppBpeHugeSamayikBmCi",
    "numTppBpeHugeOod",
    "numTppBpeHugeOodCi",
    "numTppBpeHugeOodBlockCi",
    "numTppBpeHugeFlores",
    "numBpeMonotoneSequences",
    "numBpeSequencesTotal",
    "numBpeFloresStepGap",
    "numUnigramBmPieces",
    "numUnigramHugePieces",
    # The side decomposition.
    "numCharRatioSamayik",
    "numCharRatioItihasa",
    "numCharRatioFactor",
    "numDensityRatioSamayikLo",
    "numDensityRatioSamayikHi",
    "numDensityRatioItihasaLo",
    "numDensityRatioItihasaHi",
    "numDensityRatioSamayikHugeLo",
    "numDensityRatioSamayikHugeHi",
    "numDensityPairGap",
    "numTSevenSamayik",
    "numTSevenItihasa",
    "numTSevenVocab",
    # The block bootstrap.
    "numBlockLength",
    "numBlockRowsPerCorpus",
    "numBlockWidenItihasaLo",
    "numBlockWidenItihasaHi",
    "numBlockWidenFloresLo",
    "numBlockWidenFloresHi",
    "numBlockWidenOodLo",
    "numBlockWidenOodHi",
    "numBlockWidenSamayikLo",
    "numBlockWidenSamayikHi",
    "numBlockNarrowerSamayik",
    "numBlockItihasaMaxUpper",
    "numEnTokensOTwoHundredK",
    "numEnTokensEOneBpe",
    "numEnTokenSavingPct",
    "numUnigramSixtyFourPieces",
    "numUnigramSixtyFourShortfallPct",
    "numVocabMin",
    "numVocabMax",
    "numLargeVocabFloor",
    "numVocabSmall",
    "numVocabLarge",
    "numArmsDeployedGeneral",
    "numArmsDeployedIndic",
    "numTrainedArms",
    "numCheckedSentences",
    "numLeakedSentences",
    "numBytesPerDevanagariChar",
    "numSamayikTrain",
    "numItihasaTrain",
    "numTrainPairsTotal",
    "numRenyiAlpha",
    "numRenyiDeployedLo",
    "numRenyiDeployedHi",
    "numRenyiGptTwo",
    "numRenyiGptTwoSlpOne",
    "numRenyiDeployedSlpOneLo",
    "numRenyiDeployedSlpOneHi",
    "numRenyiBpeThirtyTwo",
    "numRenyiBpeSixtyFour",
    "numRenyiUnigramLo",
    "numRenyiUnigramHi",
    "numRenyiEnglishLo",
    "numRenyiEnglishHi",
    "numLengthSparseBelow",
    "numSamayikMeanEnWords",
    "numSamayikMeanSaWords",
    "numItihasaMeanEnWords",
    "numItihasaMeanSaWords",
    "numLengthGradientsFlipped",
    "numLengthGradientsTotal",
    "numLengthEnFirstBin",
    "numLengthEnFirstLo",
    "numLengthEnFirstHi",
    "numLengthEnLastBin",
    "numLengthEnLastLo",
    "numLengthEnLastHi",
    "numLengthPairsBelowOneEn",
    "numLengthSaFirstBin",
    "numLengthSaFirstLo",
    "numLengthSaFirstHi",
    "numLengthSaLastBin",
    "numLengthSaLastLo",
    "numLengthSaLastHi",
    "numLengthPairsBelowOneSa",
    "numVerseBelowProseComparisons",
    "numVerseBelowProseTotal",
    "numVerseProseGapMin",
    "numVerseProseGapMax",
    "numEnglishBinExample",
    "numEnglishBinItihasaSaWords",
    "numEnglishBinSamayikSaWords",
    "numSanskritBinExample",
    "numSanskritBinItihasaEnWords",
    "numSanskritBinSamayikEnWords",
)


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def tex_escape(name: str) -> str:
    """Escape the one character an arm name carries that LaTeX would eat."""
    return name.replace("_", r"\_")


def arm_tt(name: str, provisional: bool = False) -> str:
    """Typeset an arm name as the experiment READMEs do, with `*` for provisional arms."""
    star = "$^{*}$" if provisional else ""
    return f"\\texttt{{{tex_escape(name)}}}{star}"


def is_provisional(name: str) -> bool:
    """True for the trained Sanskrit arms, which the READMEs mark with `*`."""
    return name in TRAINED_ARMS


def ratio(value: float) -> str:
    """A token ratio, to three decimals."""
    return f"{value:.3f}"


def two(value: float) -> str:
    """Fertility or compression, to two decimals."""
    return f"{value:.2f}"


def count(value: int) -> str:
    """A count, with thousands separators."""
    return f"{value:,}"


def math_count(value: int) -> str:
    """A count for use inside math mode, where a bare comma gets list spacing."""
    return count(value).replace(",", "{,}")


def ci(node: dict[str, Any]) -> str:
    """A bootstrap interval as `[lo, hi]`, to three decimals."""
    return f"[{float(node['ci_low']):.3f}, {float(node['ci_high']):.3f}]"


def value_ci(node: dict[str, Any], bold_below_one: bool = False) -> str:
    """`value [lo, hi]`, optionally bold when the interval sits entirely below 1.0."""
    body = f"{ratio(float(node['value']))} {ci(node)}"
    if bold_below_one and float(node["ci_high"]) < 1.0:
        return f"\\textbf{{{body}}}"
    return body


def table_float(
    body: str,
    caption: str,
    label: str,
    wide: bool = True,
    position: str = "t",
) -> str:
    """Wrap a tabular in a float, caption below the table as ACL styles it."""
    env = "table*" if wide else "table"
    return (
        f"\\begin{{{env}}}[{position}]\n"
        "\\centering\n"
        f"{body}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{{env}}}\n"
    )


def tabular(
    colspec: str,
    header: list[str],
    rows: list[list[str]],
    size: str = "\\small",
    colsep_pt: float | None = None,
) -> str:
    """A booktabs tabular. Rows are already-formatted cells.

    `colsep_pt` narrows the inter-column padding, which is the cheapest way to bring a
    wide table inside the margin without shrinking the type further. It is set inside the
    float, so it does not leak into any other table.
    """
    lines: list[str] = []
    if colsep_pt is not None:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep_pt}pt}}")
    lines.extend(
        [size, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(header) + r" \\", "\\midrule"]
    )
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Snapshot access
# --------------------------------------------------------------------------------------


def load_results(results_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the two snapshot files. Read-only: nothing here ever writes to `results/`."""
    exp01_path = results_dir / "01_baseline_penalty" / "results.json"
    exp02_path = results_dir / "02_tpp_parallel" / "results.json"
    for path in (exp01_path, exp02_path):
        if not path.exists():
            raise FileNotFoundError(f"missing snapshot file {path}")
    exp01: dict[str, Any] = json.loads(exp01_path.read_text())
    exp02: dict[str, Any] = json.loads(exp02_path.read_text())
    return exp01, exp02


def vocab_size(exp: dict[str, Any], arm: str) -> int:
    """An arm's id-space size, as the experiment recorded it."""
    return int(exp["tokenizer_sources"][arm]["vocab_size"])


def fertility_value(exp01: dict[str, Any], arm: str, language: str, variant: str) -> float:
    return float(exp01["metrics"][arm][language][variant]["fertility"]["value"])


def compression_value(exp01: dict[str, Any], arm: str, language: str, variant: str) -> float:
    return float(exp01["metrics"][arm][language][variant]["compression"]["value"])


def available_deployed(exp02: dict[str, Any]) -> list[str]:
    """Deployed arms present in this snapshot, in reporting order."""
    present = exp02["tpp"]["samayik_test"]
    return [arm for arm in (*T0_ARMS, *T3_ARMS) if arm in present]


def available_trained(exp02: dict[str, Any]) -> list[str]:
    present = exp02["tpp"]["samayik_test"]
    return [arm for arm in TRAINED_ARMS if arm in present]


def controlled_english_arms() -> tuple[str, ...]:
    """The English control arms this paper reports, from its matched pairs."""
    return tuple(dict.fromkeys(pair.split("/")[1] for pair in CONTROLLED_PAIRS))


def vocab_token(arm: str) -> str:
    """The vocabulary size an arm's name asks for: `E1_bpe_64k_bm` -> `64k`."""
    parts = arm.split("_")
    for part in reversed(parts):
        if part.endswith("k") and part[:-1].isdigit():
            return part
    raise ValueError(f"no vocabulary size in arm name {arm}")  # pragma: no cover


def requested_vocab(arm: str) -> int:
    """The id-space size an arm was asked for, from its name."""
    return int(vocab_token(arm)[:-1]) * 1000


def is_size_matched(exp02: dict[str, Any], arm: str) -> bool:
    """False when the trainer stopped short of the vocabulary size the arm asked for.

    The Unigram EM trainer stops when the corpus supports no more pieces, so two of the
    English control arms carry fewer pieces than their names promise. That breaks
    CLAUDE.md §2.5's matched-vocabulary requirement for those rows, which is why every
    table that prints them marks them and every caption states the direction of the bias.
    """
    return vocab_size(exp02, arm) == requested_vocab(arm)


def is_byte_matched(en_arm: str) -> bool:
    """True for the byte-matched control arms, whose names end in `_bm`."""
    return en_arm.endswith("_bm")


def control_label(exp02: dict[str, Any], en_arm: str) -> str:
    """`E1` or `E1\\_bm`, daggered when the arm is not size-matched."""
    label = "\\texttt{E1\\_bm}" if is_byte_matched(en_arm) else "\\texttt{E1}"
    return label + ("$^{\\ddagger}$" if not is_size_matched(exp02, en_arm) else "")


def reported_pairs(exp02: dict[str, Any], tokens: tuple[str, ...] | None = None) -> list[str]:
    """The reported matched pairs the snapshot carries, optionally at given sizes."""
    configured = {f"{pair[0]}/{pair[1]}" for pair in exp02["config"]["controlled_pairs"]}
    return [
        pair
        for pair in CONTROLLED_PAIRS
        if pair in configured
        and (tokens is None or vocab_token(pair.split("/")[0]) in tokens)
    ]


def undersized_control_arms(exp02: dict[str, Any]) -> list[str]:
    """Every English control arm the trainer could not bring to its requested size."""
    return [
        arm
        for arm in controlled_english_arms()
        if arm in exp02["tokenizer_sources"] and not is_size_matched(exp02, arm)
    ]


def reported_arms(exp02: dict[str, Any]) -> list[str]:
    """Every arm the manuscript discusses, in reporting order.

    The snapshot carries more than the paper reports: Experiment 02 keeps adding arms
    whose prose is not written yet. Enumerating the snapshot instead of this list would
    put an undiscussed arm into the arms table and the R\'enyi table.
    """
    order = (*T0_ARMS, *T3_ARMS, *TRAINED_ARMS, BYTE_ARM, *controlled_english_arms())
    return [arm for arm in order if arm in exp02["tokenizer_sources"]]


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def parity_table(exp01: dict[str, Any]) -> str:
    """Table 1: parity ratios on FLORES-200 devtest, the RQ1 headline."""
    rows: list[list[str]] = []
    for arm in T0_ARMS:
        node = exp01["parity"][arm]
        rows.append(
            [
                arm_tt(arm),
                ratio(float(node["eng_Latn"]["value"])),
                ratio(float(node["hin_Deva"]["value"])),
                ratio(float(node["eng_Latn__slp1"]["value"])),
            ]
        )
    body = tabular(
        "lrrr",
        ["Arm", "Sa/En", "Sa/Hi", "Sa (SLP1)/En"],
        rows,
        size="\\footnotesize",
    )
    n = int(exp01["n_sentences_used"])
    caption = (
        "Parity ratios on "
        f"{count(n)} aligned FLORES-200 devtest sentences: Sanskrit tokens divided by the "
        "tokens the same tokenizer spends on the English or Hindi translation of the same "
        "sentence. Sa (SLP1)/En scores the SLP1 transliteration of the Sanskrit side "
        "against the same Latin-script English pivot. These four arms are deployed "
        "practice, not a controlled comparison: their vocabularies differ by more than "
        "five times (Appendix~\\ref{sec:arms})."
    )
    return table_float(body, caption, "tab:parity", wide=False)


def fertility_compression_table(exp01: dict[str, Any]) -> str:
    """Table 2: fertility and compression, plus the ratios fertility would have implied."""
    rows: list[list[str]] = []
    for arm in T0_ARMS:
        f_sa = fertility_value(exp01, arm, "san_Deva", "original")
        f_hi = fertility_value(exp01, arm, "hin_Deva", "original")
        f_en = fertility_value(exp01, arm, "eng_Latn", "original")
        rows.append(
            [
                arm_tt(arm),
                two(f_sa),
                two(f_hi),
                two(f_en),
                two(f_sa / f_en),
                two(f_sa / f_hi),
                two(compression_value(exp01, arm, "san_Deva", "original")),
                two(compression_value(exp01, arm, "eng_Latn", "original")),
            ]
        )
    body = tabular(
        "lrrrrrrr",
        [
            "Arm",
            "Sa",
            "Hi",
            "En",
            "Sa/En",
            "Sa/Hi",
            "Sa bytes/tok.",
            "En bytes/tok.",
        ],
        rows,
    )
    caption = (
        "Fertility (tokens per whitespace word, columns 2--4) on the same sentences, the "
        "ratios that fertility implies (columns 5--6), and UTF-8 bytes per token (columns "
        "7--8). Original script throughout. Fertility is reported for comparability with "
        "the literature and is not this paper's measure of cost: Sanskrit's denominator is "
        "its word count, which sandhi and compounding make small, so columns 5--6 overstate "
        "the penalty that Table~\\ref{tab:parity} measures. Bytes per token are not "
        "comparable across scripts, since Devanagari is three UTF-8 bytes per character. "
        "The SLP1 variants of every column are in Table~\\ref{tab:fertcompfull}."
    )
    return table_float(body, caption, "tab:fertcomp", wide=True)


def fertility_compression_full_table(exp01: dict[str, Any]) -> str:
    """Appendix: fertility and compression for every arm, language and script variant."""
    rows: list[list[str]] = []
    for arm in T0_ARMS:
        for language in ("san_Deva", "hin_Deva", "eng_Latn"):
            for variant in ("original", "slp1"):
                node = exp01["metrics"][arm][language]
                if variant not in node:
                    continue
                marker = "$^{\\dagger}$" if (language == "hin_Deva" and variant == "slp1") else ""
                rows.append(
                    [
                        arm_tt(arm),
                        f"\\texttt{{{tex_escape(language)}}}",
                        f"{variant}{marker}",
                        two(float(node[variant]["fertility"]["value"])),
                        two(float(node[variant]["fertility"]["std"])),
                        two(float(node[variant]["compression"]["value"])),
                        count(int(node[variant]["fertility"]["n"])),
                    ]
                )
    body = tabular(
        "lllrrrr",
        ["Arm", "Language", "Script", "Fertility", "s.d.", "Bytes/tok.", "Words"],
        rows,
        size="\\footnotesize",
    )
    coverage = exp01["slp1_coverage"]
    hi_nukta = int(coverage["hin_Deva"]["n_with_nukta"])
    hi_nonascii = int(coverage["hin_Deva"]["n_non_ascii_after_slp1"])
    total = int(exp01["n_sentences_used"])
    caption = (
        "Fertility and compression for every arm, language and script variant. "
        "$^{\\dagger}$~The Hindi SLP1 rows are approximate and should never be quoted as a "
        "measurement of Hindi: SLP1 encodes the Sanskrit phoneme inventory, so of the "
        f"{count(total)} Hindi sentences {count(hi_nukta)} contain a nukta consonant that "
        f"the transliterator emits as a literal ASCII digit, and {count(hi_nonascii)} of "
        "the resulting strings still contain unconverted Devanagari. Both effects inflate "
        "the Hindi token count. No parity number in this paper uses an SLP1 Hindi pivot."
    )
    return table_float(body, caption, "tab:fertcompfull", wide=True)


def _deployed_rows(exp02: dict[str, Any], corpus: str) -> list[list[str]]:
    rows: list[list[str]] = []
    tpp = exp02["tpp"][corpus]
    fert = exp02["fertility"][corpus]
    byte_arm = [BYTE_ARM] if BYTE_ARM in tpp else []
    for arm in (*available_deployed(exp02), *available_trained(exp02), *byte_arm):
        slp1 = tpp[arm]["slp1"]
        original = tpp[arm].get("original")
        cells = [
            arm_tt(arm, provisional=is_provisional(arm)),
            count(vocab_size(exp02, arm)),
            value_ci(slp1["T0_o200k"], bold_below_one=True),
            value_ci(slp1["T0_llama4"], bold_below_one=True),
            value_ci(original["T0_o200k"]) if original else "n/a",
            two(float(fert[arm]["slp1"]["value"])),
        ]
        rows.append(cells)
    return rows


def _deployed_header() -> list[str]:
    return [
        "Arm",
        "Vocab.",
        "SLP1 vs \\texttt{o200k}",
        "SLP1 vs Llama-4",
        "Orig.\\ vs \\texttt{o200k}",
        "Fert.",
    ]


def tpp_deployed_all_tables(exp02: dict[str, Any]) -> str:
    """Appendix A: the deployed-practice table for every corpus, primary corpus first.

    The first of these carries the column description and the rest point at it. Deployed
    practice is reported here rather than in the body because it is not a controlled
    comparison (CLAUDE.md §2.5): vocabulary size and training domain vary alongside
    language, which is what Table \\ref{tab:tppcontrolled} holds fixed.
    """
    parts: list[str] = []
    first: str | None = None
    for corpus in exp02["tpp"]:
        body = tabular("llllll", _deployed_header(), _deployed_rows(exp02, corpus),
                       size="\\scriptsize")
        n = int(exp02["corpora"][corpus]["n_used"])
        label = f"tab:tppdeployed-{corpus.replace('_', '-')}"
        if first is None:
            caption = (
                "Deployed practice on "
                f"{CORPUS_LABELS.get(corpus, tex_escape(corpus))}, "
                f"$n={math_count(n)}$ aligned pairs. Tokens per proposition: Sanskrit "
                "tokens under the named arm divided by English tokens under a deployed "
                "200k-vocabulary English tokenizer, with 95\\% paired bootstrap "
                "intervals. This is not a controlled comparison: vocabulary size and "
                "training domain vary alongside language, which is what "
                "Table~\\ref{tab:tppcontrolled} holds fixed. $^{*}$~marks the provisional "
                f"trained arms and {arm_tt(BYTE_ARM)} the byte-level reference. Fertility "
                "is in the last column for completeness and is not part of any claim "
                "here. Bold marks an interval entirely below 1.0. "
                "\\texttt{T3\\_indicsuper} is omitted: no candidate repository resolved."
            )
            first = label
        else:
            caption = (
                f"Deployed practice on {CORPUS_LABELS.get(corpus, tex_escape(corpus))}, "
                f"$n={math_count(n)}$ pairs. Columns as in Table~\\ref{{{first}}}."
            )
        parts.append(table_float(body, caption, label, wide=True))
    return "\n".join(parts)


def tpp_controlled_table(exp02: dict[str, Any]) -> str:
    """Table 4: the matched-control TPP under both controls and all three sizes.

    Rows are grouped by algorithm and ordered by vocabulary size, so that the sweep of
    §5.3 reads down a column, and each Sanskrit arm carries two rows: its pair-matched
    control and its byte-matched one. Neither is *the* control, so neither is given the
    table to itself.
    """
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = reported_pairs(exp02)
    rows: list[list[str]] = []
    for pair in pairs:
        sa_arm, en_arm = pair.split("/")
        cells = [matched_pair_short(sa_arm), control_label(exp02, en_arm)]
        cells.extend(
            value_ci(exp02["tpp_controlled"][corpus][pair], bold_below_one=True)
            for corpus in corpora
        )
        rows.append(cells)
    header = ["Matched pair", "Control"]
    header.extend(SHORT_CORPUS_LABELS.get(corpus, tex_escape(corpus)) for corpus in corpora)
    body = tabular(
        "ll" + "l" * len(corpora), header, rows, size="\\scriptsize", colsep_pt=3.4
    )
    n_by_corpus = ", ".join(
        f"{CORPUS_LABELS.get(corpus, corpus)} {count(int(exp02['corpora'][corpus]['n_used']))}"
        for corpus in corpora
    )
    settled = sorted({vocab_size(exp02, arm) for arm in undersized_control_arms(exp02)})
    undersized = " and ".join(count(size) for size in settled)
    caption = (
        "The controlled comparison. Each row is one matched pair: a trained Sanskrit arm "
        "over an English arm sharing its algorithm, vocabulary size and training corpus. "
        "\\texttt{E1} is the pair-matched control, trained on the English side of the very "
        "sentences whose Sanskrit side trained the arm; \\texttt{E1\\_bm} is the "
        "byte-matched control, trained on a subsample of that text cut to the Sanskrit "
        "corpus's byte count. \\texttt{BPE 32k} is \\texttt{T1\\_bpe\\_raw\\_32k} over "
        "\\texttt{E1\\_bpe\\_32k}, and so on. Values are tokens per proposition with 95\\% "
        "paired bootstrap intervals, Sanskrit scored in SLP1 and English as written; bold "
        f"marks an interval entirely below 1.0. Pairs: {n_by_corpus}. "
        "$^{\\ddagger}$~marks a Unigram control the trainer could not bring to the "
        f"requested size, settling at {undersized} pieces, which makes the English side "
        "dearer and pushes those rows down."
    )
    return table_float(body, caption, "tab:tppcontrolled", wide=True)


def decomposition_table(exp02: dict[str, Any]) -> str:
    """The exact factorisation of the ratio into text length and tokens per character."""
    if "side_decomposition" not in exp02:
        return "% not available in this snapshot\n"
    corpora = (LENGTH_PROSE_CORPUS, LENGTH_VERSE_CORPUS)
    pairs = [
        *reported_pairs(exp02, SMALL_VOCAB_TOKENS),
        *[
            pair
            for pair in reported_pairs(exp02, (HUGE_VOCAB_TOKEN,))
            if "_bpe_" in pair.split("/")[0] and not is_byte_matched(pair.split("/")[1])
        ],
    ]
    rows: list[list[str]] = []
    for pair in pairs:
        sa_arm, en_arm = pair.split("/")
        cells = [matched_pair_short(sa_arm), control_label(exp02, en_arm)]
        for corpus in corpora:
            node = exp02["side_decomposition"][corpus][pair]
            cells.extend(
                ratio(float(node[key])) for key in ("char_ratio", "density_ratio", "tpp")
            )
        rows.append(cells)
    if BYTE_PAIR in exp02["side_decomposition"][corpora[0]]:
        cells = [arm_tt(BYTE_ARM), "---"]
        for corpus in corpora:
            node = exp02["side_decomposition"][corpus][BYTE_PAIR]
            cells.extend(
                ratio(float(node[key])) for key in ("char_ratio", "density_ratio", "tpp")
            )
        rows.append(cells)
    header = ["Matched pair", "Control"]
    for corpus in corpora:
        label = SHORT_CORPUS_LABELS.get(corpus, tex_escape(corpus))
        header.extend([f"{label} chars", "density", "TPP"])
    body = tabular("ll" + "r" * 6, header, rows, size="\\scriptsize", colsep_pt=4.0)
    caption = (
        "Tokens per proposition factorised exactly, on the primary prose corpus and on the "
        "verse corpus. Per corpus: the character ratio "
        "$\\sum_i c(s_i) / \\sum_i c(e_i)$, how much text each side spends on the same "
        "propositions; the density ratio, each side's tokens per character over the "
        "other's; and their product, the ratio of Table~\\ref{tab:tppcontrolled}. Sanskrit "
        "is read in SLP1, one character per phoneme, so the character ratio does not depend "
        f"on the tokenizer and repeats down its column. {arm_tt(BYTE_ARM)} has no "
        "vocabulary to vary, so its product is the text ratio itself."
    )
    return table_float(body, caption, "tab:decomposition", wide=True)


def block_ci_table(exp02: dict[str, Any]) -> str:
    """Appendix: the i.i.d. interval beside the block-resampled one, pair by corpus."""
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = [*reported_pairs(exp02)]
    if BYTE_PAIR in exp02["tpp_controlled"][corpora[0]]:
        pairs.append(BYTE_PAIR)
    sample = exp02["tpp_controlled"][corpora[0]][pairs[0]]
    if "ci_low_block" not in sample:  # pragma: no cover - not this snapshot
        return "% not available in this snapshot\n"
    header = ["Matched pair", "Control", "TPP", "i.i.d.\\ interval", "Block interval",
              "Width"]
    lines = [
        "\\setlength{\\tabcolsep}{4pt}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        " & ".join(header) + r" \\",
    ]
    for corpus in corpora:
        lines.append("\\midrule")
        label = CORPUS_LABELS.get(corpus, tex_escape(corpus))
        node0 = exp02["tpp_controlled"][corpus][pairs[0]]
        n_blocks = int(node0["n_blocks"])
        lines.append(
            f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}, "
            f"{count(int(node0['n']))} pairs, {n_blocks} blocks}} \\\\"
        )
        for pair in pairs:
            sa_arm, en_arm = pair.split("/")
            node = exp02["tpp_controlled"][corpus][pair]
            width = float(node["ci_high"]) - float(node["ci_low"])
            width_block = float(node["ci_high_block"]) - float(node["ci_low_block"])
            if pair == BYTE_PAIR:
                cells = [arm_tt(BYTE_ARM), "---"]
            else:
                cells = [matched_pair_short(sa_arm), control_label(exp02, en_arm)]
            cells.extend(
                [
                    ratio(float(node["value"])),
                    ci(node),
                    f"[{float(node['ci_low_block']):.3f}, "
                    f"{float(node['ci_high_block']):.3f}]",
                    f"{width_block / width:.2f}$\\times$",
                ]
            )
            lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    block_length = int(sample["block_length"])
    level = int(float(sample["ci"]) * 100)
    caption = (
        f"Every controlled ratio under both resampling schemes. The {level}\\% i.i.d.\\ "
        "interval resamples aligned pairs, and is the interval every other table in this "
        "paper reports; the block interval resamples non-overlapping blocks of "
        f"{block_length} consecutive pairs in corpus order, the final short block kept, "
        "and is reported because these corpora are not exchangeable sentence by sentence: "
        "Itih\\=asa test is consecutive verses of one epic and FLORES devtest consecutive "
        "sentences of the documents it was drawn from. Width is the block interval's "
        "width over the i.i.d.\\ one. Rows, controls and the $^{\\ddagger}$ mark are as in "
        f"Table~\\ref{{tab:tppcontrolled}}, with {arm_tt(BYTE_ARM)}, the byte-level "
        "reference, added at the foot of each block."
    )
    return table_float(body="\n".join(lines), caption=caption, label="tab:blockci",
                       wide=True)


def tpp_hindi_table(exp02: dict[str, Any]) -> str:
    """Appendix: the Hindi pivot on FLORES, original script and the approximate SLP1."""
    rows: list[list[str]] = []
    for arm, node in exp02["tpp_hindi"].items():
        rows.append([arm_tt(arm), count(vocab_size(exp02, arm)), value_ci(node["original"])])
    body = tabular(
        "lll",
        ["Arm", "Vocab.", "Sa/Hi (original script)"],
        rows,
        size="\\footnotesize",
    )
    caption = (
        "Sanskrit over Hindi on FLORES devtest, both sides under the same tokenizer, in "
        "the original script. The corresponding SLP1 column is computed and stored in the "
        "results file but is not printed: SLP1 encodes the Sanskrit phoneme inventory, so "
        "Hindi characters outside it inflate the Hindi token count, which sits in this "
        "ratio's denominator and mechanically depresses it. "
        "Table~\\ref{tab:fertcompfull} documents that failure and counts the sentences it "
        "affects. The trained arms are excluded by design: the Hindi pivot is defined over "
        "the deployed arms only."
    )
    return table_float(body, caption, "tab:tpphindi", wide=True)


def preregistration_table(exp01: dict[str, Any], exp02: dict[str, Any]) -> str:
    """Table 5: the pre-registered predictions of outline §8 against what was measured."""
    fert_sa = [fertility_value(exp01, arm, "san_Deva", "original") for arm in LARGE_VOCAB_T0]
    fert_hi = [fertility_value(exp01, arm, "hin_Deva", "original") for arm in LARGE_VOCAB_T0]
    parity_hi = [float(exp01["parity"][arm]["hin_Deva"]["value"]) for arm in T0_ARMS]
    t3_hi = [float(exp02["tpp_hindi"][arm]["original"]["value"]) for arm in T3_ARMS]
    t0_hi = [
        float(exp02["tpp_hindi"][arm]["original"]["value"])
        for arm in T0_ARMS
        if arm in exp02["tpp_hindi"]
    ]
    best = exp02["tpp"]["samayik_test"]["T1_bpe_raw_64k"]["slp1"]["T0_o200k"]
    best_ctrl = exp02["tpp_controlled"]["samayik_test"]["T1_bpe_raw_64k/E1_bpe_64k"]
    huge_ctrl = exp02["tpp_controlled"]["samayik_test"]["T1_bpe_raw_128k/E1_bpe_128k"]
    huge_ood = exp02["tpp_controlled"]["samayik_test_ood"]["T1_bpe_raw_128k/E1_bpe_128k"]

    # The Hindi band was pre-registered as [2, 4]. Count the arms that fall outside it
    # rather than asserting "held": one arm below the floor is not the same as holding.
    below_floor = [value for value in fert_hi if value < PREREG_HINDI_FERTILITY_LO]
    above_ceiling = [value for value in fert_hi if value > PREREG_HINDI_FERTILITY_HI]
    if not below_floor and not above_ceiling:  # pragma: no cover - not this snapshot
        hindi_outcome = "Held"
    else:
        crossed = "lower bound" if below_floor else "upper bound"
        n_crossing = len(below_floor) or len(above_ceiling)
        arms = "arm" if n_crossing == 1 else "arms"
        hindi_outcome = (
            f"Approximately held; {crossed} crossed by {n_crossing} {arms} of "
            f"{len(fert_hi)}"
        )

    # The T3 prediction assumed a Sanskrit-over-Hindi gap that T3 would narrow. The T3
    # arms sit inside or above the T0 range rather than below it, so the outcome states
    # that rather than denying the gap: T0's own range is well clear of 1.0. "Within or
    # above" is the accurate reading, since the lowest T3 arm sits inside the T0 range
    # rather than at its floor.
    t3_outcome = (
        "Not observed; T3 arms sit within or above the T0 range"
        if min(t3_hi) >= min(t0_hi)
        else "Not observed"  # pragma: no cover - not this snapshot
    )

    rows = [
        [
            "T0 fertility on Sanskrit "
            f"{PREREG_FERTILITY_LO}--{PREREG_FERTILITY_HI}",
            f"{two(min(fert_sa))}--{two(max(fert_sa))} at $\\geq$200k vocab; "
            f"{two(fertility_value(exp01, 'T0_gpt2', 'san_Deva', 'original'))} for GPT-2",
            "Refuted, except GPT-2 on Devanagari",
        ],
        [
            f"Hindi {PREREG_HINDI_FERTILITY_LO}--{PREREG_HINDI_FERTILITY_HI} "
            "under the same tokenizer",
            f"{two(min(fert_hi))}--{two(max(fert_hi))} at $\\geq$200k vocab",
            hindi_outcome,
        ],
        [
            f"Sa/Hi parity $>{PREREG_PARITY_THRESHOLD}$",
            f"{ratio(min(parity_hi))}--{ratio(max(parity_hi))}",
            "Refuted",
        ],
        [
            "T3 closes most of the gap vs Hindi",
            f"Sa/Hi {ratio(min(t3_hi))}--{ratio(max(t3_hi))} for T3 against "
            f"{ratio(min(t0_hi))}--{ratio(max(t0_hi))} for T0",
            t3_outcome,
        ],
        [
            "TPP crosses below 1.0 against \\texttt{o200k}, for the proposed "
            "\\texttt{T6} arm",
            "\\texttt{T6} is not built here. On the raw-subword baselines, the only "
            "Sanskrit-native arms this paper builds: "
            f"{ratio(float(best['value']))} {ci(best)} on prose against the deployed "
            f"pivot; {ratio(float(best_ctrl['value']))} {ci(best_ctrl)} under the matched "
            f"control at {count(vocab_size(exp02, 'T1_bpe_raw_64k'))} pieces, and "
            f"{ratio(float(huge_ctrl['value']))} {ci(huge_ctrl)} at "
            f"{count(requested_vocab('T1_bpe_raw_128k'))} on in-domain prose against "
            f"{ratio(float(huge_ood['value']))} {ci(huge_ood)} out of domain",
            "Observed against the pivot; under the matched control only at "
            f"{count(requested_vocab('T1_bpe_raw_128k'))} in domain",
        ],
    ]
    body = tabular(
        "p{0.26\\textwidth}p{0.42\\textwidth}p{0.24\\textwidth}",
        ["Pre-registered prediction", "Measured", "Outcome"],
        rows,
    )
    caption = (
        "The predictions of our design document against what was measured. They were "
        "written before any arm was run and are dated only by our repository's history: "
        "this is a design document we did not edit, not a registration with an external "
        "timestamp, and it is reported for that reason and no stronger one. The first "
        "four are guesses at magnitudes rather than substantive hypotheses. The fifth was "
        "written for a sandhi-split, morpheme-constrained arm this paper does not build, "
        "so its Outcome is recorded against the raw-subword baselines this paper does "
        "build instead: the crossing is real against the deployed English pivot, and "
        "against the matched English control it appears only at the largest vocabulary "
        "trained here and only on in-domain prose."
    )
    return table_float(body, caption, "tab:prereg", wide=True)


def arms_table(exp01: dict[str, Any], exp02: dict[str, Any]) -> str:
    """Appendix B: what each arm actually loaded, and its id-space size."""
    rows: list[list[str]] = []
    seen: set[str] = set()
    for arm in reported_arms(exp02):
        node = exp02["tokenizer_sources"][arm]
        seen.add(arm)
        source = str(node["source_id"])
        sha = str(node.get("sha256", ""))
        detail = f"\\texttt{{{tex_escape(source)}}}"
        if sha:
            detail = f"trained here; \\texttt{{tokenizer.json}} sha256 \\texttt{{{sha[:12]}}}"
        rows.append(
            [
                arm_tt(arm, provisional=is_provisional(arm)),
                str(node.get("family", "")),
                count(int(node["vocab_size"])),
                detail,
            ]
        )
    rows.sort(key=lambda row: (row[1], row[0]))
    for arm in exp01["tokenizer_sources"]:
        if arm not in seen:  # pragma: no cover - the two snapshots share their T0 arms
            node = exp01["tokenizer_sources"][arm]
            rows.append([arm_tt(arm), "T0", count(int(node["vocab_size"])),
                         f"\\texttt{{{tex_escape(str(node['source_id']))}}}"])
    body = tabular(
        "llrl",
        ["Arm", "Family", "Vocab.", "What loaded"],
        rows,
        size="\\footnotesize",
    )
    unavailable = "; ".join(
        f"\\texttt{{{tex_escape(arm)}}}" for arm in exp02.get("unavailable_arms", {})
    )
    caption = (
        "Every arm, the identifier that actually loaded, and its id-space size "
        "($\\mathrm{len}(\\mathrm{tokenizer})$, which counts added special tokens). "
        "\\texttt{T0\\_llama4} and \\texttt{T0\\_gemma3} load ungated re-uploads of the "
        "official releases because the official repositories are gated and the machine "
        "running these experiments has no access token; the vocabulary sizes match the "
        "published ones, and the mirrors could not be byte-compared against the originals "
        "because reading the originals is what the gate prevents. A Hub repository is "
        "mutable and these rows carry no revision hash, so the identifier names what was "
        "loaded and not which bytes: the load date is the \\texttt{timestamp} field of "
        "the results file the table is generated from. The sha256 discipline applies only "
        "to the arms trained here, whose \\texttt{tokenizer.json} this repository holds. "
        f"Unavailable this run: {unavailable}."
    )
    return table_float(body, caption, "tab:arms", wide=True)


def matched_pair_short(sa_arm: str) -> str:
    """`T1_bpe_raw_32k` -> `BPE 32k`. The caption spells the pairs out in full."""
    algorithm = "BPE" if "_bpe_" in sa_arm else "Unigram"
    return f"{algorithm} {sa_arm.rsplit('_', 1)[-1]}"


def bin_label_tex(name: str) -> str:
    """A bin name as the prose sets it: `25-40` -> `25--40`, `41+` unchanged."""
    return name.replace("-", "--")


def controlled_pair_keys(exp02: dict[str, Any]) -> list[str]:
    """The matched pairs the length blocks carry.

    Restricted to `LENGTH_PAIRS`, which is what Experiment 02 stratifies: its
    `length_strata_skip_arms` leaves the 128k arms, the byte-matched controls and the byte
    reference out of the strata, and a table must not print a number the snapshot does not
    hold or the manuscript does not discuss.
    """
    configured = {f"{pair[0]}/{pair[1]}" for pair in exp02["config"]["controlled_pairs"]}
    return [pair for pair in LENGTH_PAIRS if pair in configured]


def _length_cell(node: dict[str, Any] | None, suppress_below: int | None = None) -> str:
    """`value [lo, hi]`, daggered when the bin is sparse.

    Nothing here is bolded. Marking the intervals below 1.0 in a table whose own caption
    says the gradient is a selection artefact sends two signals at once, so the reading is
    left to the prose. With `suppress_below` set, a bin holding fewer than that many pairs
    prints the dagger alone: the ratio stays in the appendix table, where the reader has
    already been told the bin is thin.
    """
    if node is None or node.get("value") is None:
        return "n/a"
    sparse = bool(node.get("sparse", False))
    if suppress_below is not None and int(node.get("n_pairs", 0)) < suppress_below:
        return "$^{\\dagger}$"
    text = f"{ratio(float(node['value']))} {ci(node)}"
    if sparse:
        text += "$^{\\dagger}$"
    return text


def _length_block(
    strata: dict[str, Any],
    corpus: str,
    pairs: list[str],
    bins: list[str],
    suppress_below: int | None = None,
) -> list[list[str]]:
    """One corpus inside a length table: a heading, the bins' `n`, then the four pairs.

    `n` is a row rather than a per-cell annotation because it is a property of the bin,
    not of the pair: all four matched pairs score the same sentences.
    """
    width = len(bins) + 1
    heading = CORPUS_LABELS.get(corpus, tex_escape(corpus))
    rows: list[list[str]] = [[f"\\multicolumn{{{width}}}{{l}}{{\\emph{{{heading}}}}}"]]
    counts = ["$n$ pairs"]
    for name in bins:
        node = strata[corpus][pairs[0]].get(name)
        if node is None or node.get("value") is None:
            counts.append("--")
            continue
        cell = count(int(node["n_pairs"]))
        counts.append(f"{cell}$^{{\\dagger}}$" if node.get("sparse") else cell)
    rows.append(counts)
    for pair in pairs:
        cells = [matched_pair_short(pair.split("/")[0])]
        cells.extend(
            _length_cell(strata[corpus][pair].get(name), suppress_below) for name in bins
        )
        rows.append(cells)
    return rows


def _length_tabular(header: list[str], blocks: list[list[list[str]]]) -> str:
    """A booktabs tabular whose corpus blocks are separated by their own rules."""
    lines = [
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\scriptsize",
        "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(header) + r" \\",
    ]
    for block in blocks:
        lines.append("\\midrule")
        lines.extend(" & ".join(row) + r" \\" for row in block)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _length_caption(
    exp02: dict[str, Any],
    side: str,
    edges_key: str,
    tail: str,
    terse: bool = False,
) -> str:
    """The appendix length captions: what is binned, and how to read a cell.

    `terse` drops the reading instructions and points at the half of the body table that
    carries them; repeating eight lines of boilerplate a page later costs a column and
    tells the reader nothing new. The body table writes its own caption, since it holds
    both stratifications and has to say so once.
    """
    edges = ", ".join(str(edge) for edge in exp02["config"][edges_key])
    level = int(float(exp02["config"].get("ci", 0.95)) * 100)
    floor = int(exp02["config"]["length_sparse_below"])
    script = " word count in the original script" if side == "Sanskrit" else " word count"
    opening = (
        "Tokens per proposition under the matched control, stratified by the "
        f"\\textbf{{{side}}} side's whitespace{script} (bin edges {edges}). "
    )
    if terse:
        return (
            opening + "Cells, rows and daggers are as in "
            f"Table~\\ref{{tab:tppbylength}}. {tail}"
        )
    return (
        opening
        + f"Each cell is the ratio with its {level}\\% paired bootstrap interval; nothing "
        "is bolded, since which intervals clear 1.0 is read in the prose. Rows are the "
        "matched pairs of Table~\\ref{tab:tppcontrolled} in short form (BPE 32k is "
        "\\texttt{T1\\_bpe\\_raw\\_32k} over \\texttt{E1\\_bpe\\_32k}, and so on), and all "
        "four score the same sentences, so the $n$ row belongs to the bin. "
        f"$^{{\\dagger}}$~marks a bin with fewer than {floor} pairs. {tail}"
    )


def tpp_by_length_table(exp02: dict[str, Any]) -> str:
    """Both stratifications of the two primary corpora, in one float, prose first.

    One table rather than two, because the section's whole argument is that neither
    stratification is read alone: putting the mirror on the facing page under its own
    caption invited exactly the single-table reading the text warns against, and cost a
    column of caption boilerplate that said the same thing twice.

    Restricted to the matched pairs, since a length breakdown of the uncontrolled
    deployed-practice ratios would answer a question this paper does not ask
    (CLAUDE.md §2.5).
    """
    if "tpp_by_length" not in exp02 or "tpp_by_length_sa" not in exp02:
        return "% not available in this snapshot\n"
    pairs = controlled_pair_keys(exp02)
    lines: list[str] = [
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
    ]
    for index, (key, _edges_key, side) in enumerate(LENGTH_STRATA):
        strata: dict[str, Any] = exp02[key]
        bins = list(strata[LENGTH_BODY_CORPORA[0]][pairs[0]].keys())
        abbreviation = "Sa" if side == "Sanskrit" else "En"
        letter = "ab"[index]
        if index:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{len(bins) + 1}}}{{l}}{{\\emph{{({letter}) Bins cut on the "
            f"{side} side}}}} \\\\"
        )
        header = ["Matched pair"]
        header.extend(f"{bin_label_tex(name)} {abbreviation}.\\ words" for name in bins)
        lines.append(" & ".join(header) + r" \\")
        for corpus in LENGTH_BODY_CORPORA:
            lines.append("\\midrule")
            block = _length_block(
                strata, corpus, pairs, bins, LENGTH_BODY_SUPPRESS_BELOW
            )
            lines.extend(" & ".join(row) + r" \\" for row in block)
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    level = int(float(exp02["config"].get("ci", 0.95)) * 100)
    floor = int(exp02["config"]["length_sparse_below"])
    edges_en = ", ".join(str(edge) for edge in exp02["config"]["length_bin_edges"])
    edges_sa = ", ".join(str(edge) for edge in exp02["config"]["length_bin_edges_sa"])
    caption = (
        "Tokens per proposition under the matched control, stratified by sentence length: "
        f"(a) on the English side's whitespace word count (bin edges {edges_en}), (b) on "
        f"the Sanskrit side's, in the original script (bin edges {edges_sa}). Cells are "
        f"the ratio with its {level}\\% paired bootstrap interval, and nothing is bolded, "
        "since which intervals clear 1.0 is read in the prose. Rows are the matched pairs "
        "of Table~\\ref{tab:tppcontrolled} (BPE 32k is \\texttt{T1\\_bpe\\_raw\\_32k} over "
        "\\texttt{E1\\_bpe\\_32k}); all four score the same sentences, so the $n$ row "
        f"belongs to the bin. $^{{\\dagger}}$~marks a bin under {floor} pairs; under "
        f"{LENGTH_BODY_SUPPRESS_BELOW} it stands alone, the ratio left to "
        "Table~\\ref{tab:tppbylengthall}. The halves are read together: binning on the "
        "English side selects pairs whose English side is long for their content, and that "
        "side is the denominator, so (a)'s gradient runs down whether or not density "
        "changes with length, while in (b) the binned side is the numerator and it runs "
        "up. The other two corpora are in Appendix~\\ref{sec:lengthstrata}."
    )
    return table_float("\n".join(lines), caption, "tab:tppbylength", wide=True)


def tpp_by_length_all_tables(exp02: dict[str, Any]) -> str:
    """Appendix: every corpus under both stratifications, as two floats."""
    if "tpp_by_length" not in exp02:
        return "% not available in this snapshot\n"
    pairs = controlled_pair_keys(exp02)
    parts: list[str] = []
    for key, edges_key, side in LENGTH_STRATA:
        if key not in exp02:  # pragma: no cover - both keys are in this snapshot
            continue
        strata: dict[str, Any] = exp02[key]
        corpora = list(strata.keys())
        bins = list(strata[corpora[0]][pairs[0]].keys())
        blocks = [_length_block(strata, corpus, pairs, bins) for corpus in corpora]
        abbreviation = "Sa" if side == "Sanskrit" else "En"
        header = ["Matched pair"]
        header.extend(f"{bin_label_tex(name)} {abbreviation}.\\ words" for name in bins)
        tail = (
            "Every corpus, at the same bins as the corresponding half of the body's "
            "Table~\\ref{tab:tppbylength}, which carries the two primary corpora. Read "
            "against its companion on the other side: "
            "a gradient that keeps its sign under both stratifications would be "
            "consistent with a length effect and one that changes sign is selection, "
            "and here all "
            "\\numLengthGradientsTotal{} change sign."
        )
        caption = _length_caption(
            exp02, side, edges_key, tail, terse=(side == "Sanskrit")
        )
        label = "tab:tppbylengthall" + ("sa" if side == "Sanskrit" else "")
        parts.append(table_float(_length_tabular(header, blocks), caption, label, wide=True))
    return "\n".join(parts)


def renyi_table(exp02: dict[str, Any], corpus: str = "samayik_test") -> str:
    """Renyi efficiency on the primary prose corpus, when the snapshot carries it."""
    if "renyi" not in exp02:
        return "% not available in this snapshot\n"
    renyi: dict[str, Any] = exp02["renyi"][corpus]
    english: dict[str, Any] = exp02.get("renyi_english", {}).get(corpus, {})
    alphas = [str(float(alpha)) for alpha in exp02["config"]["renyi_alphas"]]

    rows: list[list[str]] = []
    for arm in (*T0_ARMS, *T3_ARMS, *RENYI_TRAINED_ARMS):
        node = renyi.get(arm)
        if node is None:
            continue
        cells = [arm_tt(arm, provisional=is_provisional(arm))]
        for variant in ("original", "slp1"):
            for alpha in alphas:
                entry = node.get(variant, {}).get(alpha)
                cells.append(ratio(float(entry["value"])) if entry else "n/a")
        rows.append(cells)
    for arm in (*T0_ARMS, *RENYI_ENGLISH_ARMS):
        node = english.get(arm)
        if node is None:
            continue
        cells = [f"{arm_tt(arm)} (English side)"]
        cells.extend("n/a" for _ in alphas)
        cells.extend(
            ratio(float(node[alpha]["value"])) if alpha in node else "n/a" for alpha in alphas
        )
        rows.append(cells)

    header = ["Arm"]
    header.extend(f"orig.\\ $\\alpha={alpha}$" for alpha in alphas)
    header.extend(f"SLP1 $\\alpha={alpha}$" for alpha in alphas)
    body = tabular("l" + "r" * (2 * len(alphas)), header, rows, size="\\footnotesize")
    caption = (
        "R\\'enyi efficiency on "
        f"{CORPUS_LABELS.get(corpus, tex_escape(corpus))}, Sanskrit side in both script "
        "variants. The English-side rows carry the English text under the named pivot and "
        "have no script variant, so their values are placed in the right-hand pair of "
        "columns. Reported as a diagnostic only: the measure can be gamed, so it supports "
        "no claim in this paper on its own, and it is tabulated for the size-matched "
        f"{count(vocab_size(exp02, 'T1_bpe_raw_32k'))}- and "
        f"{count(vocab_size(exp02, 'T1_bpe_raw_64k'))}-piece arms and their pair-matched "
        "controls rather than extended over the byte-matched and "
        f"{count(requested_vocab('T1_bpe_raw_128k'))}-piece arms of "
        "Section~\\ref{sec:rq2-controlled}."
    )
    return table_float(body, caption, "tab:renyi", wide=True)


# --------------------------------------------------------------------------------------
# Prose macros
# --------------------------------------------------------------------------------------


def renyi_macros(exp02: dict[str, Any]) -> dict[str, str]:
    """The macros §5.6 reads, at one α on the primary prose corpus.

    Returns an empty mapping when the snapshot carries no `renyi` block, in which case the
    caller's completeness check reports the missing names. GPT-2 is separated from the
    other deployed arms exactly as it is in §5.1: its vocabulary predates any serious
    Devanagari coverage, so it is a period piece rather than part of the deployed band.
    """
    if "renyi" not in exp02:
        return {}
    alpha = RENYI_PROSE_ALPHA
    renyi = exp02["renyi"][RENYI_PROSE_CORPUS]
    english = exp02.get("renyi_english", {}).get(RENYI_PROSE_CORPUS, {})

    def sanskrit(arm: str, variant: str) -> float:
        return float(renyi[arm][variant][alpha]["value"])

    band = [arm for arm in (*T0_ARMS, *T3_ARMS) if arm in renyi and arm != "T0_gpt2"]
    original = [sanskrit(arm, "original") for arm in band]
    slp1 = [sanskrit(arm, "slp1") for arm in band]
    english_values = [
        float(english[arm][alpha]["value"])
        for arm in (*T0_ARMS, *RENYI_ENGLISH_ARMS)
        if arm in english
    ]
    unigram = [
        sanskrit(arm, "slp1") for arm in ("T2_unigram_raw_32k", "T2_unigram_raw_64k")
    ]
    return {
        "numRenyiAlpha": alpha,
        "numRenyiDeployedLo": ratio(min(original)),
        "numRenyiDeployedHi": ratio(max(original)),
        "numRenyiDeployedSlpOneLo": ratio(min(slp1)),
        "numRenyiDeployedSlpOneHi": ratio(max(slp1)),
        "numRenyiGptTwo": ratio(sanskrit("T0_gpt2", "original")),
        "numRenyiGptTwoSlpOne": ratio(sanskrit("T0_gpt2", "slp1")),
        "numRenyiBpeThirtyTwo": ratio(sanskrit("T1_bpe_raw_32k", "slp1")),
        "numRenyiBpeSixtyFour": ratio(sanskrit("T1_bpe_raw_64k", "slp1")),
        "numRenyiUnigramLo": ratio(min(unigram)),
        "numRenyiUnigramHi": ratio(max(unigram)),
        "numRenyiEnglishLo": ratio(min(english_values)),
        "numRenyiEnglishHi": ratio(max(english_values)),
    }


def _dense_bins(entries: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The populated, non-sparse bins of one corpus x pair, in bin order."""
    return [
        (name, node)
        for name, node in entries.items()
        if node.get("value") is not None and not node.get("sparse", False)
    ]


def _gradient_sign(entries: dict[str, Any]) -> float | None:
    """Last dense bin minus first dense bin, the gradient this section reads."""
    dense = _dense_bins(entries)
    if len(dense) < 2:
        return None
    return float(dense[-1][1]["value"]) - float(dense[0][1]["value"])


def length_gradient_flips(exp02: dict[str, Any]) -> tuple[int, int]:
    """How many corpus x pair gradients change sign between the two stratifications."""
    pairs = controlled_pair_keys(exp02)
    flipped = 0
    total = 0
    for corpus in exp02["tpp_by_length"]:
        for pair in pairs:
            english = _gradient_sign(exp02["tpp_by_length"][corpus][pair])
            sanskrit = _gradient_sign(exp02["tpp_by_length_sa"][corpus][pair])
            if english is None or sanskrit is None:  # pragma: no cover - not this snapshot
                continue
            total += 1
            if (english > 0) != (sanskrit > 0):
                flipped += 1
    return flipped, total


def verse_below_prose(exp02: dict[str, Any]) -> tuple[int, int, float, float]:
    """Verse against prose at every bin both corpora populate, under both stratifications.

    A comparison is one bin x one matched pair x one stratification, counted only where
    both corpora hold at least `length_sparse_below` pairs in that bin, so that neither
    side of the comparison rests on a handful of sentences. Returns the number of those
    in which verse sits below prose, the total, and the smallest and largest gap.
    """
    pairs = controlled_pair_keys(exp02)
    floor = int(exp02["config"]["length_sparse_below"])
    below = 0
    total = 0
    gaps: list[float] = []
    for key, _edges_key, _side in LENGTH_STRATA:
        strata = exp02[key]
        prose = strata[LENGTH_PROSE_CORPUS]
        verse = strata[LENGTH_VERSE_CORPUS]
        for name in prose[pairs[0]]:
            if int(prose[pairs[0]][name]["n_pairs"]) < floor:
                continue
            if int(verse[pairs[0]][name]["n_pairs"]) < floor:
                continue
            for pair in pairs:
                total += 1
                gap = float(prose[pair][name]["value"]) - float(verse[pair][name]["value"])
                gaps.append(gap)
                if gap > 0:
                    below += 1
    return below, total, min(gaps), max(gaps)


def _corpus_mean_words(exp02: dict[str, Any], corpus: str) -> tuple[float, float]:
    """A corpus's mean English and Sanskrit word counts, pooled over its English bins."""
    pair = controlled_pair_keys(exp02)[0]
    entries = exp02["tpp_by_length"][corpus][pair]
    total = 0
    english = 0.0
    sanskrit = 0.0
    for node in entries.values():
        n = int(node["n_pairs"])
        if n == 0:  # pragma: no cover - every bin is populated in this snapshot
            continue
        total += n
        english += n * float(node["mean_words_en"])
        sanskrit += n * float(node["mean_words_sa"])
    return english / total, sanskrit / total


def _extreme_dense_bin(
    exp02: dict[str, Any], key: str, corpus: str, last: bool
) -> tuple[str, list[float]]:
    """The first or last dense bin of `corpus`, and its value under each matched pair."""
    pairs = controlled_pair_keys(exp02)
    dense = _dense_bins(exp02[key][corpus][pairs[0]])
    name = dense[-1][0] if last else dense[0][0]
    return name, [float(exp02[key][corpus][pair][name]["value"]) for pair in pairs]


def _busiest_joint_bin(exp02: dict[str, Any], key: str) -> str:
    """The jointly populated bin holding the most verse pairs, under one stratification.

    Jointly populated means both corpora clear `length_sparse_below` there, which is the
    same floor the verse-against-prose count uses.
    """
    pair = controlled_pair_keys(exp02)[0]
    floor = int(exp02["config"]["length_sparse_below"])
    prose = exp02[key][LENGTH_PROSE_CORPUS][pair]
    verse = exp02[key][LENGTH_VERSE_CORPUS][pair]
    joint: list[str] = [
        str(name)
        for name in prose
        if int(prose[name]["n_pairs"]) >= floor and int(verse[name]["n_pairs"]) >= floor
    ]
    return max(joint, key=lambda name: int(verse[name]["n_pairs"]))


def length_macros(exp02: dict[str, Any]) -> dict[str, str]:
    """The macros §5.5 reads. Empty when the snapshot predates the length strata."""
    if "tpp_by_length" not in exp02 or "tpp_by_length_sa" not in exp02:
        return {}
    values: dict[str, str] = {}
    values["numLengthSparseBelow"] = str(int(exp02["config"]["length_sparse_below"]))

    for key, corpus in (
        ("numSamayik", LENGTH_PROSE_CORPUS),
        ("numItihasa", LENGTH_VERSE_CORPUS),
    ):
        english, sanskrit = _corpus_mean_words(exp02, corpus)
        values[f"{key}MeanEnWords"] = f"{english:.1f}"
        values[f"{key}MeanSaWords"] = f"{sanskrit:.1f}"

    flipped, total = length_gradient_flips(exp02)
    values["numLengthGradientsFlipped"] = str(flipped)
    values["numLengthGradientsTotal"] = str(total)

    for prefix, key in (("numLengthEn", "tpp_by_length"), ("numLengthSa", "tpp_by_length_sa")):
        for position, last in (("First", False), ("Last", True)):
            name, bin_values = _extreme_dense_bin(exp02, key, LENGTH_PROSE_CORPUS, last)
            values[f"{prefix}{position}Bin"] = bin_label_tex(name)
            values[f"{prefix}{position}Lo"] = ratio(min(bin_values))
            values[f"{prefix}{position}Hi"] = ratio(max(bin_values))
    # The prose says where the prose corpus crosses under each stratification: the last
    # dense English bin, and the first dense Sanskrit bin, are the crossings that exist.
    _, en_last = _extreme_dense_bin(exp02, "tpp_by_length", LENGTH_PROSE_CORPUS, True)
    _, sa_first = _extreme_dense_bin(exp02, "tpp_by_length_sa", LENGTH_PROSE_CORPUS, False)
    values["numLengthPairsBelowOneEn"] = str(sum(1 for value in en_last if value < 1.0))
    values["numLengthPairsBelowOneSa"] = str(sum(1 for value in sa_first if value < 1.0))

    below, comparisons, gap_min, gap_max = verse_below_prose(exp02)
    values["numVerseBelowProseComparisons"] = str(below)
    values["numVerseBelowProseTotal"] = str(comparisons)
    values["numVerseProseGapMin"] = ratio(gap_min)
    values["numVerseProseGapMax"] = ratio(gap_max)

    # The two bin-mean examples: a bin equates the corpora on the side it is cut on and
    # leaves the other side free, which is why the separation is bracketed, not isolated.
    # The illustrating bin is the jointly populated one holding the most verse pairs,
    # under each stratification, so the example is the bulk of the verse corpus.
    pair = controlled_pair_keys(exp02)[0]
    en_bin_name = _busiest_joint_bin(exp02, "tpp_by_length")
    values["numEnglishBinExample"] = bin_label_tex(en_bin_name)
    for label, corpus in (
        ("numEnglishBinItihasa", LENGTH_VERSE_CORPUS),
        ("numEnglishBinSamayik", LENGTH_PROSE_CORPUS),
    ):
        node = exp02["tpp_by_length"][corpus][pair][en_bin_name]
        values[f"{label}SaWords"] = f"{float(node['mean_words_sa']):.1f}"
    sa_bin_name = _busiest_joint_bin(exp02, "tpp_by_length_sa")
    values["numSanskritBinExample"] = bin_label_tex(sa_bin_name)
    for label, corpus in (
        ("numSanskritBinItihasa", LENGTH_VERSE_CORPUS),
        ("numSanskritBinSamayik", LENGTH_PROSE_CORPUS),
    ):
        node = exp02["tpp_by_length_sa"][corpus][pair][sa_bin_name]
        values[f"{label}EnWords"] = f"{float(node['mean_words_en']):.1f}"
    return values


def _megabytes(value: int) -> str:
    """A corpus size in megabytes, to two decimals, as the Limitations section reads it."""
    return f"{value / 1_000_000:.2f}"


def byte_matched_macros(exp02: dict[str, Any]) -> dict[str, str]:
    """The macros §5.3's byte-matched paragraph reads.

    The move is measured pair by pair: for each Sanskrit arm and corpus, the ratio against
    the pair-matched control against the ratio against the byte-matched one. A verdict is
    whether the interval sits above 1.0, below it, or straddles it, and the count of
    verdicts that change between the two controls is what the paragraph claims.
    """
    controlled = exp02["tpp_controlled"]
    moves: list[float] = []
    changed = 0
    for corpus in controlled:
        for pair in reported_pairs(exp02):
            sa_arm, en_arm = pair.split("/")
            if is_byte_matched(en_arm):
                continue
            twin = f"{sa_arm}/{en_arm}_bm"
            if twin not in controlled[corpus]:  # pragma: no cover - not this snapshot
                continue
            node = controlled[corpus][pair]
            twin_node = controlled[corpus][twin]
            moves.append(float(twin_node["value"]) - float(node["value"]))
            if _verdict(node) != _verdict(twin_node):
                changed += 1
    median = statistics.median(abs(move) for move in moves)
    excess = ENGLISH_TRAIN_BYTES / SANSKRIT_TRAIN_BYTES - 1.0
    return {
        "numBmLines": count(ENGLISH_BM_TRAIN_LINES),
        "numBmLinePct": f"{ENGLISH_BM_TRAIN_LINES / ENGLISH_TRAIN_LINES * 100:.1f}",
        "numBytesSanskritTrain": _megabytes(SANSKRIT_TRAIN_BYTES),
        "numBytesEnglishTrain": _megabytes(ENGLISH_TRAIN_BYTES),
        "numBytesEnglishBmTrain": _megabytes(ENGLISH_BM_TRAIN_BYTES),
        "numBytesEnglishExcessPct": f"{excess * 100:.0f}",
        "numBmMoveCount": str(len(moves)),
        "numBmMaxMove": f"{max(abs(move) for move in moves):.3f}",
        "numBmMedianMove": f"{median:.3f}",
        "numBmDownwardCount": str(sum(1 for move in moves if move < 0)),
        "numBmVerdictChanges": str(changed),
    }


def _verdict(node: dict[str, Any]) -> str:
    """Where an interval sits relative to parity: above it, below it, or across it."""
    if float(node["ci_low"]) > 1.0:
        return "above"
    if float(node["ci_high"]) < 1.0:
        return "below"
    return "straddles"  # pragma: no cover - no controlled interval straddles i.i.d. here


def bpe_progression(exp02: dict[str, Any]) -> dict[str, list[float]]:
    """The BPE ratio at each vocabulary size, per corpus and per control.

    Keyed `corpus/E1` and `corpus/E1_bm`, values in `VOCAB_TOKENS` order. This is the
    sweep §5.3 reads: whether the controlled penalty shrinks as the vocabulary grows.
    """
    controlled = exp02["tpp_controlled"]
    series: dict[str, list[float]] = {}
    for corpus in controlled:
        for suffix, name in (("", "E1"), ("_bm", "E1_bm")):
            values: list[float] = []
            for token in VOCAB_TOKENS:
                pair = f"T1_bpe_raw_{token}/E1_bpe_{token}{suffix}"
                if pair not in controlled[corpus]:  # pragma: no cover - not this snapshot
                    break
                values.append(float(controlled[corpus][pair]["value"]))
            if len(values) == len(VOCAB_TOKENS):
                series[f"{corpus}/{name}"] = values
    return series


def vocabulary_macros(exp02: dict[str, Any]) -> dict[str, str]:
    """The macros §5.3's vocabulary paragraph reads."""
    controlled = exp02["tpp_controlled"]
    series = bpe_progression(exp02)
    monotone = sum(
        1 for values in series.values() if values[0] > values[1] > values[2]
    )
    # The one sequence that is not monotone at every step: FLORES under the pair-matched
    # control, where the first two sizes are within a thousandth of each other. The prose
    # states that gap rather than rounding it away.
    steps = [
        abs(values[1] - values[0])
        for values in series.values()
        if not values[0] > values[1] > values[2]
    ]
    values: dict[str, str] = {
        "numVocabHuge": count(requested_vocab("T1_bpe_raw_128k")),
        "numBpeMonotoneSequences": str(monotone),
        "numBpeSequencesTotal": str(len(series)),
        "numBpeFloresStepGap": f"{max(steps):.3f}" if steps else "0.000",
    }
    for key, corpus, pair in (
        ("numTppBpeHugeSamayik", "samayik_test", "T1_bpe_raw_128k/E1_bpe_128k"),
        ("numTppBpeHugeSamayikBm", "samayik_test", "T1_bpe_raw_128k/E1_bpe_128k_bm"),
        ("numTppBpeHugeOod", "samayik_test_ood", "T1_bpe_raw_128k/E1_bpe_128k"),
    ):
        node = controlled[corpus][pair]
        values[key] = ratio(float(node["value"]))
        values[f"{key}Ci"] = ci(node)
    ood = controlled["samayik_test_ood"]["T1_bpe_raw_128k/E1_bpe_128k"]
    values["numTppBpeHugeOodBlockCi"] = (
        f"[{float(ood['ci_low_block']):.3f}, {float(ood['ci_high_block']):.3f}]"
    )
    values["numTppBpeHugeFlores"] = ratio(
        float(controlled["flores_devtest"]["T1_bpe_raw_128k/E1_bpe_128k"]["value"])
    )
    return values


def decomposition_macros(exp02: dict[str, Any]) -> dict[str, str]:
    """The macros §5.4 reads: the character ratio, the density band, the byte reference."""
    if "side_decomposition" not in exp02:  # pragma: no cover - not this snapshot
        return {}
    decomposition = exp02["side_decomposition"]
    small = reported_pairs(exp02, SMALL_VOCAB_TOKENS)
    huge = reported_pairs(exp02, (HUGE_VOCAB_TOKEN,))
    prose = decomposition[LENGTH_PROSE_CORPUS]
    verse = decomposition[LENGTH_VERSE_CORPUS]
    char_prose = float(prose[small[0]]["char_ratio"])
    char_verse = float(verse[small[0]]["char_ratio"])
    values = {
        "numCharRatioSamayik": ratio(char_prose),
        "numCharRatioItihasa": ratio(char_verse),
        "numCharRatioFactor": ratio(char_prose / char_verse),
        "numDensityPairGap": ratio(
            max(
                abs(float(prose[pair]["density_ratio"]) - float(verse[pair]["density_ratio"]))
                for pair in small
            )
        ),
        "numTSevenSamayik": ratio(float(prose[BYTE_PAIR]["tpp"])),
        "numTSevenItihasa": ratio(float(verse[BYTE_PAIR]["tpp"])),
        "numTSevenVocab": count(vocab_size(exp02, BYTE_ARM)),
    }
    # The 128k band the prose quotes is the size-matched BPE pair under both controls:
    # it is read beside the crossing, and the Unigram rows at that size are not
    # size-matched and do not cross.
    huge_bpe = [pair for pair in huge if "_bpe_" in pair.split("/")[0]]
    for key, node, pairs in (
        ("numDensityRatioSamayik", prose, small),
        ("numDensityRatioItihasa", verse, small),
        ("numDensityRatioSamayikHuge", prose, huge_bpe),
    ):
        band = [float(node[pair]["density_ratio"]) for pair in pairs]
        values[f"{key}Lo"] = ratio(min(band))
        values[f"{key}Hi"] = ratio(max(band))
    return values


def block_macros(exp02: dict[str, Any]) -> dict[str, str]:
    """The macros §3 and the block-interval appendix read."""
    controlled = exp02["tpp_controlled"]
    pairs = [*reported_pairs(exp02)]
    if BYTE_PAIR in controlled[LENGTH_PROSE_CORPUS]:
        pairs.append(BYTE_PAIR)
    sample = controlled[LENGTH_PROSE_CORPUS][pairs[0]]
    values = {
        "numBlockLength": str(int(sample["block_length"])),
        "numBlockRowsPerCorpus": str(len(pairs)),
    }
    for key, corpus in (
        ("numBlockWidenSamayik", "samayik_test"),
        ("numBlockWidenOod", "samayik_test_ood"),
        ("numBlockWidenItihasa", "itihasa_test"),
        ("numBlockWidenFlores", "flores_devtest"),
    ):
        widths = []
        for pair in pairs:
            node = controlled[corpus][pair]
            iid = float(node["ci_high"]) - float(node["ci_low"])
            block = float(node["ci_high_block"]) - float(node["ci_low_block"])
            widths.append(block / iid)
        values[f"{key}Lo"] = f"{min(widths):.1f}"
        values[f"{key}Hi"] = f"{max(widths):.1f}"
    narrower = sum(
        1
        for pair in pairs
        if (
            float(controlled["samayik_test"][pair]["ci_high_block"])
            - float(controlled["samayik_test"][pair]["ci_low_block"])
        )
        < (
            float(controlled["samayik_test"][pair]["ci_high"])
            - float(controlled["samayik_test"][pair]["ci_low"])
        )
    )
    values["numBlockNarrowerSamayik"] = str(narrower)
    values["numBlockItihasaMaxUpper"] = ratio(
        max(float(controlled["itihasa_test"][pair]["ci_high_block"]) for pair in pairs)
    )
    return values


def numbers_macros(exp01: dict[str, Any], exp02: dict[str, Any]) -> str:
    """Every number the manuscript's prose contains, as a `\\newcommand`."""
    values: dict[str, str] = {}

    # Corpora and protocol.
    values["numFloresSents"] = count(int(exp01["n_sentences_used"]))
    for key, corpus in (
        ("numSamayikTest", "samayik_test"),
        ("numSamayikOod", "samayik_test_ood"),
        ("numItihasaTest", "itihasa_test"),
    ):
        values[key] = count(int(exp02["corpora"][corpus]["n_used"]))
    sample = exp02["tpp_controlled"]["samayik_test"]["T1_bpe_raw_64k/E1_bpe_64k"]
    values["numBootstrap"] = str(int(sample["n_bootstrap"]))
    values["numBootstrapSeed"] = str(int(sample["seed"]))
    values["numCiLevel"] = f"{float(sample['ci']) * 100:.0f}"

    # RQ1: words, parity, fertility, compression.
    values["numSaWords"] = count(
        int(exp01["metrics"]["T0_o200k"]["san_Deva"]["original"]["fertility"]["n"])
    )
    values["numEnWords"] = count(
        int(exp01["metrics"]["T0_o200k"]["eng_Latn"]["original"]["fertility"]["n"])
    )
    values["numHiWords"] = count(
        int(exp01["metrics"]["T0_o200k"]["hin_Deva"]["original"]["fertility"]["n"])
    )
    parity_en = [float(exp01["parity"][arm]["eng_Latn"]["value"]) for arm in LARGE_VOCAB_T0]
    values["numParitySaEnLo"] = ratio(min(parity_en))
    values["numParitySaEnHi"] = ratio(max(parity_en))
    values["numParitySaEnGptTwo"] = ratio(float(exp01["parity"]["T0_gpt2"]["eng_Latn"]["value"]))
    parity_hi = [float(exp01["parity"][arm]["hin_Deva"]["value"]) for arm in T0_ARMS]
    values["numParitySaHiLo"] = ratio(min(parity_hi))
    values["numParitySaHiHi"] = ratio(max(parity_hi))
    # The abstract states the Sanskrit/Hindi range over the same arms as the
    # Sanskrit/English one beside it, which is the large-vocabulary subset only; the
    # four-arm range above belongs to Section 5.1, where GPT-2 has just been named.
    parity_hi_large = [
        float(exp01["parity"][arm]["hin_Deva"]["value"]) for arm in LARGE_VOCAB_T0
    ]
    values["numParitySaHiLargeLo"] = ratio(min(parity_hi_large))
    values["numParitySaHiLargeHi"] = ratio(max(parity_hi_large))

    fert_sa = [fertility_value(exp01, arm, "san_Deva", "original") for arm in LARGE_VOCAB_T0]
    values["numFertSaLo"] = two(min(fert_sa))
    values["numFertSaHi"] = two(max(fert_sa))
    values["numFertSaGptTwo"] = two(fertility_value(exp01, "T0_gpt2", "san_Deva", "original"))
    values["numFertSaGptTwoSlpOne"] = two(fertility_value(exp01, "T0_gpt2", "san_Deva", "slp1"))
    ratio_en = [
        fertility_value(exp01, arm, "san_Deva", "original")
        / fertility_value(exp01, arm, "eng_Latn", "original")
        for arm in T0_ARMS
    ]
    ratio_hi = [
        fertility_value(exp01, arm, "san_Deva", "original")
        / fertility_value(exp01, arm, "hin_Deva", "original")
        for arm in T0_ARMS
    ]
    values["numFertRatioSaEnLo"] = two(min(ratio_en))
    values["numFertRatioSaEnHi"] = two(max(ratio_en))
    values["numFertRatioSaHiLo"] = two(min(ratio_hi))
    values["numFertRatioSaHiHi"] = two(max(ratio_hi))

    comp_sa = [compression_value(exp01, arm, "san_Deva", "original") for arm in T0_ARMS]
    comp_en = [compression_value(exp01, arm, "eng_Latn", "original") for arm in T0_ARMS]
    comp_sa_slp1 = [compression_value(exp01, arm, "san_Deva", "slp1") for arm in T0_ARMS]
    values["numCompSaOrigLo"] = two(min(comp_sa))
    values["numCompSaOrigHi"] = two(max(comp_sa))
    values["numCompEnLo"] = two(min(comp_en))
    values["numCompEnHi"] = two(max(comp_en))
    values["numCompSaSlpLo"] = two(min(comp_sa_slp1))
    values["numCompSaSlpHi"] = two(max(comp_sa_slp1))
    values["numBytesPerDevanagariChar"] = "three"

    # The pre-registration, quoted from the design document.
    values["numPreregFertLo"] = str(PREREG_FERTILITY_LO)
    values["numPreregFertHi"] = str(PREREG_FERTILITY_HI)
    values["numPreregHindiFertLo"] = str(PREREG_HINDI_FERTILITY_LO)
    values["numPreregHindiFertHi"] = str(PREREG_HINDI_FERTILITY_HI)
    values["numPreregParityThreshold"] = PREREG_PARITY_THRESHOLD

    # RQ2: deployed practice.
    deployed = [
        float(exp02["tpp"][corpus][arm]["slp1"]["T0_o200k"]["value"])
        for corpus in ("samayik_test", "samayik_test_ood", "flores_devtest")
        for arm in available_deployed(exp02)
    ]
    values["numDeployedLo"] = ratio(min(deployed))
    values["numDeployedHi"] = ratio(max(deployed))
    node = exp02["tpp"]["samayik_test"]["T0_o200k"]["slp1"]["T0_o200k"]
    values["numDeployedSamayikOTwoHundredK"] = ratio(float(node["value"]))
    values["numDeployedSamayikOTwoHundredKCi"] = ci(node)
    node = exp02["tpp"]["samayik_test"]["T1_bpe_raw_64k"]["slp1"]["T0_o200k"]
    values["numTppBpeSixtyFourDeployed"] = ratio(float(node["value"]))
    values["numTppBpeSixtyFourDeployedCi"] = ci(node)
    values["numEnTokensOTwoHundredK"] = count(int(node["pivot_tokens"]))

    # RQ2: the matched control.
    controlled = exp02["tpp_controlled"]
    best = controlled["samayik_test"]["T1_bpe_raw_64k/E1_bpe_64k"]
    values["numTppControlledBest"] = ratio(float(best["value"]))
    values["numTppControlledBestCi"] = ci(best)
    values["numEnTokensEOneBpe"] = count(int(best["pivot_tokens"]))
    saving = 1.0 - float(best["pivot_tokens"]) / float(node["pivot_tokens"])
    values["numEnTokenSavingPct"] = f"{saving * 100:.1f}"
    # The prose-corpus statement of §5.3 is about the sizes at which both sides are
    # size-matched under both controls, so these ranges run over those eight pairs; the
    # 128k pairs, where the in-domain verdict changes, carry their own macros below.
    small_pairs = reported_pairs(exp02, SMALL_VOCAB_TOKENS)
    huge_pairs = reported_pairs(exp02, (HUGE_VOCAB_TOKEN,))
    for key, corpus in (
        ("numTppControlledSamayik", "samayik_test"),
        ("numTppControlledOod", "samayik_test_ood"),
        ("numTppControlledItihasa", "itihasa_test"),
        ("numTppControlledFlores", "flores_devtest"),
    ):
        pairs = [float(controlled[corpus][pair]["value"]) for pair in small_pairs]
        values[f"{key}Lo"] = ratio(min(pairs))
        values[f"{key}Hi"] = ratio(max(pairs))
    every = [
        float(controlled["itihasa_test"][pair]["value"])
        for pair in reported_pairs(exp02)
    ]
    values["numTppControlledItihasaAllLo"] = ratio(min(every))
    values["numTppControlledItihasaAllHi"] = ratio(max(every))
    # Out of domain for both sides: the out-of-domain prose split and FLORES.
    out_of_domain = [
        float(controlled[corpus][pair]["value"])
        for corpus in ("samayik_test_ood", "flores_devtest")
        for pair in huge_pairs
    ]
    values["numTppControlledHugeOodLo"] = ratio(min(out_of_domain))
    values["numTppControlledHugeOodHi"] = ratio(max(out_of_domain))
    values["numTppControlledBpeThirtyTwo"] = ratio(
        float(controlled["samayik_test"]["T1_bpe_raw_32k/E1_bpe_32k"]["value"])
    )
    values["numControlledPairs"] = str(len(controlled_pair_keys(exp02)))
    values["numControlledPairsAll"] = str(len(reported_pairs(exp02)))
    values["numControlledPairsSmall"] = str(len(small_pairs))
    values["numControlledPairsHuge"] = str(len(huge_pairs))
    actual = vocab_size(exp02, "E1_unigram_64k")
    values["numUnigramSixtyFourPieces"] = count(actual)
    values["numUnigramSixtyFourShortfallPct"] = (
        f"{(1 - actual / requested_vocab('E1_unigram_64k')) * 100:.1f}"
    )
    values["numUnigramBmPieces"] = count(vocab_size(exp02, "E1_unigram_64k_bm"))
    values["numUnigramHugePieces"] = count(vocab_size(exp02, "E1_unigram_128k"))

    values.update(byte_matched_macros(exp02))
    values.update(vocabulary_macros(exp02))
    values.update(decomposition_macros(exp02))
    values.update(block_macros(exp02))

    # Setup.
    deployed_vocabs = [vocab_size(exp02, arm) for arm in available_deployed(exp02)]
    values["numVocabMin"] = count(min(deployed_vocabs))
    values["numVocabMax"] = count(max(deployed_vocabs))
    values["numLargeVocabFloor"] = count(min(vocab_size(exp02, arm) for arm in LARGE_VOCAB_T0))
    values["numVocabSmall"] = count(vocab_size(exp02, "T1_bpe_raw_32k"))
    values["numVocabLarge"] = count(vocab_size(exp02, "T1_bpe_raw_64k"))
    values["numArmsDeployedGeneral"] = str(len([a for a in available_deployed(exp02)
                                                if a.startswith("T0_")]))
    values["numArmsDeployedIndic"] = str(len([a for a in available_deployed(exp02)
                                              if a.startswith("T3_")]))
    values["numTrainedArms"] = str(len(available_trained(exp02)))
    values["numCheckedSentences"] = count(
        sum(int(entry["n"]) for entry in exp02["exclusion_check"].values())
    )
    values["numLeakedSentences"] = str(
        sum(int(entry["n_missing"]) for entry in exp02["exclusion_check"].values())
        + sum(int(entry["n_missing"]) for entry in exp02["exclusion_check_en"].values())
    )

    # Training-split sizes, from `data/README.md` rather than from the snapshot: see the
    # comment on SAMAYIK_TRAIN_PAIRS.
    values["numSamayikTrain"] = count(SAMAYIK_TRAIN_PAIRS)
    values["numItihasaTrain"] = count(ITIHASA_TRAIN_PAIRS)
    values["numTrainPairsTotal"] = count(SAMAYIK_TRAIN_PAIRS + ITIHASA_TRAIN_PAIRS)

    values.update(renyi_macros(exp02))
    values.update(length_macros(exp02))

    missing = [name for name in MACROS if name not in values]
    if missing:
        raise RuntimeError(f"MACROS lists names this writer does not emit: {missing}")
    extra = [name for name in values if name not in MACROS]
    if extra:
        raise RuntimeError(f"this writer emits names MACROS does not list: {extra}")

    lines = [
        "% Generated by scripts/paper_tables.py from results/. Do not edit.",
        "% Every number in the manuscript's prose is one of these macros.",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{values[name]}}}" for name in MACROS)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------


def build(exp01: dict[str, Any], exp02: dict[str, Any]) -> dict[str, str]:
    """Every output file, as name -> content. Pure: no I/O happens here."""
    return {
        "parity.tex": parity_table(exp01),
        "fertility_compression.tex": fertility_compression_table(exp01),
        "fertility_compression_full.tex": fertility_compression_full_table(exp01),
        "tpp_deployed_all.tex": tpp_deployed_all_tables(exp02),
        "tpp_controlled.tex": tpp_controlled_table(exp02),
        "decomposition.tex": decomposition_table(exp02),
        "block_ci.tex": block_ci_table(exp02),
        "tpp_hindi.tex": tpp_hindi_table(exp02),
        "preregistration.tex": preregistration_table(exp01, exp02),
        "arms.tex": arms_table(exp01, exp02),
        "tpp_by_length.tex": tpp_by_length_table(exp02),
        "tpp_by_length_all.tex": tpp_by_length_all_tables(exp02),
        "renyi.tex": renyi_table(exp02),
        "numbers.tex": numbers_macros(exp01, exp02),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="the tracked results snapshot to read (default: results/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "paper" / "1a" / "tables",
        help="where the .tex files go (default: paper/1a/tables)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    exp01, exp02 = load_results(args.results_dir)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in build(exp01, exp02).items():
        (out_dir / name).write_text(content)
        LOGGER.info("wrote %s", out_dir / name)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
