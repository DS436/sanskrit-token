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

Two writers (`tpp_by_length.tex`, `renyi.tex`) emit a one-line LaTeX comment when the
snapshot has no `tpp_by_length` / `renyi` key, so this script already runs against the
future snapshot that will carry them.

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
            "Held",
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
            "Not observed; no gap to close",
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


def tpp_by_length_table(exp02: dict[str, Any]) -> str:
    """Length-stratified TPP, when the snapshot carries it.

    Restricted to the matched pairs, since a length breakdown of the uncontrolled
    deployed-practice ratios would answer a question this paper does not ask.
    """
    if "tpp_by_length" not in exp02:
        return "% not available in this snapshot\n"
    strata: dict[str, Any] = exp02["tpp_by_length"]
    controlled = [f"{pair[0]}/{pair[1]}" for pair in exp02["config"]["controlled_pairs"]]
    corpora = list(strata.keys())
    bins = list(strata[corpora[0]][controlled[0]].keys())
    rows: list[list[str]] = []
    sparse_seen = False
    for corpus in corpora:
        for pair in controlled:
            sa_arm, en_arm = pair.split("/")
            cells = [
                SHORT_CORPUS_LABELS.get(corpus, tex_escape(corpus)),
                f"{arm_tt(sa_arm, provisional=True)} / {arm_tt(en_arm)}",
            ]
            for name in bins:
                node = strata[corpus][pair].get(name)
                if node is None or node.get("value") is None:
                    cells.append("n/a")
                    continue
                sparse = bool(node.get("sparse", False))
                sparse_seen = sparse_seen or sparse
                text = ratio(float(node["value"]))
                if float(node["ci_high"]) < 1.0:
                    text = f"\\textbf{{{text}}}"
                cells.append(f"{text}$^{{\\dagger}}$" if sparse else text)
            rows.append(cells)
    header = ["Corpus", "Matched pair"]
    header.extend(f"{tex_escape(name)} w." for name in bins)
    body = tabular("ll" + "r" * len(bins), header, rows, size="\\scriptsize")
    edges = ", ".join(str(edge) for edge in exp02["config"]["length_bin_edges"])
    caption = (
        "Tokens per proposition under the matched control, stratified by the number of "
        "words in the English side of the pair. Bin edges: "
        f"{edges}. Values only, to three decimals; bold marks a bin whose "
        f"{int(float(exp02['config'].get('ci', 0.95)) * 100)}\\% bootstrap interval lies "
        "entirely below 1.0, and the intervals themselves are in the snapshot's "
        "\\texttt{results.json}."
    )
    if sparse_seen:
        caption += (
            " $^{\\dagger}$~marks a bin the experiment flagged as sparse (fewer than "
            f"{int(exp02['config']['length_sparse_below'])} pairs)."
        )
    return table_float(body, caption, "tab:tppbylength", wide=True)


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
