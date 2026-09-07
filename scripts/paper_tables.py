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
TRAINED_ARMS = ("T1_bpe_raw_32k", "T1_bpe_raw_64k", "T2_unigram_raw_32k", "T2_unigram_raw_64k")
#: Arms whose ≥200k vocabulary makes them current practice rather than a period piece.
LARGE_VOCAB_T0 = ("T0_o200k", "T0_llama4", "T0_gemma3")

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
    "numControlledPairs",
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


def _deployed_rows(
    exp02: dict[str, Any], corpus: str, second_pivot: bool = True
) -> list[list[str]]:
    rows: list[list[str]] = []
    tpp = exp02["tpp"][corpus]
    fert = exp02["fertility"][corpus]
    for arm in (*available_deployed(exp02), *available_trained(exp02)):
        slp1 = tpp[arm]["slp1"]
        original = tpp[arm].get("original")
        cells = [
            arm_tt(arm, provisional=is_provisional(arm)),
            count(vocab_size(exp02, arm)),
            value_ci(slp1["T0_o200k"], bold_below_one=True),
        ]
        if second_pivot:
            cells.append(value_ci(slp1["T0_llama4"], bold_below_one=True))
        cells.append(value_ci(original["T0_o200k"]) if original else "n/a")
        cells.append(two(float(fert[arm]["slp1"]["value"])))
        rows.append(cells)
    return rows


def _deployed_header(second_pivot: bool = True) -> list[str]:
    header = ["Arm", "Vocab.", "SLP1 vs \\texttt{o200k}"]
    if second_pivot:
        header.append("SLP1 vs Llama-4")
    header.extend(["Orig.\\ vs \\texttt{o200k}", "Fert."])
    return header


def tpp_deployed_table(exp02: dict[str, Any], corpus: str = "samayik_test") -> str:
    """Table 3: deployed-practice TPP on the primary prose corpus."""
    body = tabular(
        "lllll",
        _deployed_header(second_pivot=False),
        _deployed_rows(exp02, corpus, second_pivot=False),
        size="\\footnotesize",
    )
    n = int(exp02["corpora"][corpus]["n_used"])
    caption = (
        "Deployed practice on S\\=amayik test (prose, primary; "
        f"$n={math_count(n)}$ aligned pairs). Tokens per proposition: Sanskrit tokens under "
        "the "
        "named arm divided by English tokens under a deployed 200k-vocabulary English "
        "tokenizer, with 95\\% paired bootstrap intervals. This is not a controlled "
        "comparison: vocabulary size and training domain vary alongside language, which is "
        "what Table~\\ref{tab:tppcontrolled} holds fixed. $^{*}$~marks the provisional "
        "trained arms. Fertility is in the last column for completeness and is not part of "
        "any claim here. Bold marks an interval entirely below 1.0. The second English "
        "pivot, and the other three corpora, are in Appendix~\\ref{sec:fulltables}. "
        "\\texttt{T3\\_indicsuper} is omitted: no candidate repository resolved."
    )
    return table_float(body, caption, "tab:tppdeployed", wide=True)


def tpp_deployed_all_tables(exp02: dict[str, Any]) -> str:
    """Appendix: the deployed-practice table for every corpus."""
    parts: list[str] = []
    for corpus in exp02["tpp"]:
        body = tabular("llllll", _deployed_header(), _deployed_rows(exp02, corpus),
                       size="\\scriptsize")
        n = int(exp02["corpora"][corpus]["n_used"])
        caption = (
            f"Deployed practice on {CORPUS_LABELS.get(corpus, tex_escape(corpus))}, "
            f"$n={math_count(n)}$ pairs. Columns as in Table~\\ref{{tab:tppdeployed}}."
        )
        parts.append(table_float(body, caption, f"tab:tppdeployed-{corpus.replace('_', '-')}",
                                 wide=True))
    return "\n".join(parts)


def tpp_controlled_table(exp02: dict[str, Any]) -> str:
    """Table 4: the matched-control TPP, the paper's headline."""
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = list(exp02["tpp_controlled"][corpora[0]].keys())
    rows: list[list[str]] = []
    for pair in pairs:
        sa_arm, en_arm = pair.split("/")
        cells = [arm_tt(sa_arm, provisional=True)]
        cells.extend(
            value_ci(exp02["tpp_controlled"][corpus][pair], bold_below_one=True)
            for corpus in corpora
        )
        rows.append(cells)
    twins = ", ".join(
        f"{arm_tt(pair.split('/')[0])} over {arm_tt(pair.split('/')[1])}" for pair in pairs
    )
    header = ["Sanskrit arm, over its matched English control"]
    header.extend(SHORT_CORPUS_LABELS.get(corpus, tex_escape(corpus)) for corpus in corpora)
    body = tabular(
        "l" + "l" * len(corpora), header, rows, size="\\scriptsize", colsep_pt=4.0
    )
    n_by_corpus = ", ".join(
        f"{CORPUS_LABELS.get(corpus, corpus)} {count(int(exp02['corpora'][corpus]['n_used']))}"
        for corpus in corpora
    )
    caption = (
        "The controlled comparison. Each row is one matched pair: a trained Sanskrit arm "
        "over the English arm sharing its algorithm, its vocabulary size and its training "
        "corpus (the two sides of the same S\\=amayik and Itih\\=asa training splits). "
        f"The pairs are {twins}. "
        "Values are tokens per proposition with 95\\% paired bootstrap intervals; bold "
        "marks an interval entirely below 1.0. Sanskrit is scored in SLP1, English as "
        f"written, and each arm's name carries its vocabulary size. Pairs: {n_by_corpus}. "
        "Only the verse corpus stays below parity, and "
        "verse carries a meter confound and a 19th-century English translation in the "
        "denominator. Both sides of a pair were asked for the same vocabulary size; "
        "\\texttt{E1\\_unigram\\_64k} settles on "
        f"{count(vocab_size(exp02, 'E1_unigram_64k'))} pieces, because the EM trainer "
        "stops short when the corpus does not support the full vocabulary. That shortfall "
        "makes the English side dearer, so it pushes the last row down, against the "
        "direction that row is read for."
    )
    return table_float(body, caption, "tab:tppcontrolled", wide=True)


def tpp_hindi_table(exp02: dict[str, Any]) -> str:
    """Appendix: the Hindi pivot on FLORES, original script and the approximate SLP1."""
    rows: list[list[str]] = []
    for arm, node in exp02["tpp_hindi"].items():
        rows.append(
            [
                arm_tt(arm),
                count(vocab_size(exp02, arm)),
                value_ci(node["original"]),
                value_ci(node["slp1"]) + "$^{\\dagger}$",
            ]
        )
    body = tabular(
        "llll",
        ["Arm", "Vocab.", "Sa/Hi (original)", "Sa/Hi (SLP1)"],
        rows,
        size="\\footnotesize",
    )
    caption = (
        "Sanskrit over Hindi on FLORES devtest, both sides under the same tokenizer. "
        "$^{\\dagger}$~The SLP1 column is an artifact, not a result: Hindi characters "
        "outside SLP1's Sanskrit inventory inflate the Hindi token count, which sits in "
        "this ratio's denominator and mechanically depresses it. Read the original-script "
        "column. The trained arms are excluded by design: the Hindi pivot is defined over "
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
    # arms sit at or above the T0 range, so the outcome states that rather than denying
    # the gap: T0's own range is well clear of 1.0.
    t3_outcome = (
        "Not observed; T3 arms sit at or above the T0 range"
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
            "TPP crosses below 1.0 against \\texttt{o200k}",
            f"{ratio(float(best['value']))} {ci(best)} on prose; "
            f"{ratio(float(best_ctrl['value']))} {ci(best_ctrl)} under the matched control",
            "Observed against the pivot, not under control",
        ],
    ]
    body = tabular(
        "p{0.26\\textwidth}p{0.42\\textwidth}p{0.24\\textwidth}",
        ["Pre-registered prediction", "Measured", "Outcome"],
        rows,
    )
    caption = (
        "The pre-registered predictions of the project's design document against what was "
        "measured. The predictions were written before any arm was run and are dated in "
        "the repository's history. The last row is the paper's central negative result: "
        "the crossing is real against the deployed English pivot and disappears against "
        "the matched English control, so it is a statement about the pivot's vocabulary "
        "and domain rather than about Sanskrit."
    )
    return table_float(body, caption, "tab:prereg", wide=True)


def arms_table(exp01: dict[str, Any], exp02: dict[str, Any]) -> str:
    """Appendix B: what each arm actually loaded, and its id-space size."""
    rows: list[list[str]] = []
    seen: set[str] = set()
    for arm, node in exp02["tokenizer_sources"].items():
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
        "because reading the originals is what the gate prevents. Unavailable this run: "
        f"{unavailable}."
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
    """The four matched pairs, as they key the by-length blocks."""
    return [f"{pair[0]}/{pair[1]}" for pair in exp02["config"]["controlled_pairs"]]


def _length_cell(node: dict[str, Any] | None) -> str:
    """`value [lo, hi]`, bold when the interval is below 1.0, daggered when sparse."""
    if node is None or node.get("value") is None:
        return "n/a"
    text = f"{ratio(float(node['value']))} {ci(node)}"
    if float(node["ci_high"]) < 1.0:
        text = f"\\textbf{{{text}}}"
    if bool(node.get("sparse", False)):
        text += "$^{\\dagger}$"
    return text


def _length_block(
    strata: dict[str, Any], corpus: str, pairs: list[str], bins: list[str]
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
        cells.extend(_length_cell(strata[corpus][pair].get(name)) for name in bins)
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


def _length_caption(exp02: dict[str, Any], side: str, edges_key: str, tail: str) -> str:
    """The shared body of both length captions: what is binned, and how to read a cell."""
    edges = ", ".join(str(edge) for edge in exp02["config"][edges_key])
    level = int(float(exp02["config"].get("ci", 0.95)) * 100)
    floor = int(exp02["config"]["length_sparse_below"])
    script = " word count in the original script" if side == "Sanskrit" else " word count"
    return (
        "Tokens per proposition under the matched control, stratified by the "
        f"\\textbf{{{side}}} side's whitespace{script} (bin edges {edges}). Each cell is "
        f"the ratio with its {level}\\% paired bootstrap interval; bold marks an interval "
        "entirely below 1.0. Rows name the matched pairs of "
        "Table~\\ref{tab:tppcontrolled} in short form: BPE 32k is "
        "\\texttt{T1\\_bpe\\_raw\\_32k} over \\texttt{E1\\_bpe\\_32k}, Unigram 64k is "
        "\\texttt{T2\\_unigram\\_raw\\_64k} over \\texttt{E1\\_unigram\\_64k}, and so on; "
        "all four pairs score the same sentences, so the $n$ row belongs to the bin. "
        f"$^{{\\dagger}}$~marks a bin with fewer than {floor} pairs. {tail}"
    )


def tpp_by_length_table(exp02: dict[str, Any]) -> str:
    """The English-binned strata for the two primary corpora, prose first.

    Restricted to the matched pairs, since a length breakdown of the uncontrolled
    deployed-practice ratios would answer a question this paper does not ask
    (CLAUDE.md §2.5). The companion Sanskrit-binned table is the point: binning on one
    side biases the ratio in that side's direction, so neither table is read alone.
    """
    if "tpp_by_length" not in exp02:
        return "% not available in this snapshot\n"
    strata: dict[str, Any] = exp02["tpp_by_length"]
    pairs = controlled_pair_keys(exp02)
    bins = list(strata[LENGTH_BODY_CORPORA[0]][pairs[0]].keys())
    blocks = [_length_block(strata, corpus, pairs, bins) for corpus in LENGTH_BODY_CORPORA]
    header = ["Matched pair"]
    header.extend(f"{bin_label_tex(name)} En.\\ words" for name in bins)
    tail = (
        "Selecting pairs by a high English word count preferentially selects pairs whose "
        "English side is long for its content, and that side is this ratio's denominator, "
        "so the gradient across these columns runs downwards whether or not density "
        "changes with length. Table~\\ref{tab:tppbylengthsa} is the mirror. The other two "
        "corpora are in Appendix~\\ref{sec:lengthstrata}."
    )
    caption = _length_caption(exp02, "English", "length_bin_edges", tail)
    return table_float(_length_tabular(header, blocks), caption, "tab:tppbylength", wide=True)


def tpp_by_length_sa_table(exp02: dict[str, Any]) -> str:
    """The Sanskrit-binned strata for the same two corpora, on the same pairs."""
    if "tpp_by_length_sa" not in exp02:
        return "% not available in this snapshot\n"
    strata: dict[str, Any] = exp02["tpp_by_length_sa"]
    pairs = controlled_pair_keys(exp02)
    bins = list(strata[LENGTH_BODY_CORPORA[0]][pairs[0]].keys())
    blocks = [_length_block(strata, corpus, pairs, bins) for corpus in LENGTH_BODY_CORPORA]
    header = ["Matched pair"]
    header.extend(f"{bin_label_tex(name)} Sa.\\ words" for name in bins)
    tail = (
        "The bias runs the other way here: the binned side is now the numerator, so the "
        "gradient across these columns runs upwards for the same reason "
        "Table~\\ref{tab:tppbylength}'s runs downwards. Every within-corpus gradient "
        "changes sign between the two, which is what selection on the binned side looks "
        "like; the verse-below-prose ordering does not. The other two corpora are in "
        "Appendix~\\ref{sec:lengthstrata}."
    )
    caption = _length_caption(exp02, "Sanskrit", "length_bin_edges_sa", tail)
    return table_float(
        _length_tabular(header, blocks), caption, "tab:tppbylengthsa", wide=True
    )


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
            "Every corpus, at the same bins as the body's "
            f"Table~\\ref{{tab:tppbylength{'sa' if side == 'Sanskrit' else ''}}}, which "
            "carries the two primary ones. Read against its companion on the other side: "
            "a gradient that keeps its sign under both stratifications would be "
            "consistent with a length effect and one that changes sign is selection, "
            "and here all "
            "\\numLengthGradientsTotal{} change sign."
        )
        caption = _length_caption(exp02, side, edges_key, tail)
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
    for arm, node in renyi.items():
        cells = [arm_tt(arm, provisional=is_provisional(arm))]
        for variant in ("original", "slp1"):
            for alpha in alphas:
                entry = node.get(variant, {}).get(alpha)
                cells.append(ratio(float(entry["value"])) if entry else "n/a")
        rows.append(cells)
    for arm, node in english.items():
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
        "no claim in this paper on its own."
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
    english_values = [float(node[alpha]["value"]) for node in english.values()]
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
    for key, corpus in (
        ("numTppControlledSamayik", "samayik_test"),
        ("numTppControlledOod", "samayik_test_ood"),
        ("numTppControlledItihasa", "itihasa_test"),
        ("numTppControlledFlores", "flores_devtest"),
    ):
        pairs = [float(entry["value"]) for entry in controlled[corpus].values()]
        values[f"{key}Lo"] = ratio(min(pairs))
        values[f"{key}Hi"] = ratio(max(pairs))
    values["numTppControlledBpeThirtyTwo"] = ratio(
        float(controlled["samayik_test"]["T1_bpe_raw_32k/E1_bpe_32k"]["value"])
    )
    values["numControlledPairs"] = str(len(controlled["samayik_test"]))
    requested = 64000
    actual = vocab_size(exp02, "E1_unigram_64k")
    values["numUnigramSixtyFourPieces"] = count(actual)
    values["numUnigramSixtyFourShortfallPct"] = f"{(1 - actual / requested) * 100:.1f}"

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
        "tpp_deployed.tex": tpp_deployed_table(exp02),
        "tpp_deployed_all.tex": tpp_deployed_all_tables(exp02),
        "tpp_controlled.tex": tpp_controlled_table(exp02),
        "tpp_hindi.tex": tpp_hindi_table(exp02),
        "preregistration.tex": preregistration_table(exp01, exp02),
        "arms.tex": arms_table(exp01, exp02),
        "tpp_by_length.tex": tpp_by_length_table(exp02),
        "tpp_by_length_sa.tex": tpp_by_length_sa_table(exp02),
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
