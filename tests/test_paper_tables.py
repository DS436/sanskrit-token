"""The generated tables and figures of `paper/1a` agree with the tracked snapshot.

Everything here runs offline against `results/`, which is read and never written, and it
checks the two properties that make the manuscript trustworthy: the numbers printed in the
tables are the numbers in `results.json` at the printed precision, and every number in the
prose comes from a macro the generator actually emits.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT / "paper" / "1a"

EXPECTED_TABLES = (
    "parity.tex",
    "fertility_compression.tex",
    "fertility_compression_full.tex",
    "tpp_deployed.tex",
    "tpp_deployed_all.tex",
    "tpp_controlled.tex",
    "tpp_hindi.tex",
    "preregistration.tex",
    "arms.tex",
    "tpp_by_length.tex",
    "tpp_by_length_all.tex",
    "renyi.tex",
    "numbers.tex",
)

EXPECTED_FIGURES = (
    "parity.pdf",
    "parity.png",
    "denominator.pdf",
    "denominator.png",
    "length.pdf",
    "length.png",
)

#: A table row, as the generator writes it: cells joined by `&`, terminated by `\\`.
ROW = re.compile(r"^\\texttt\{(?P<arm>[^}]+)\}(?P<rest>.*?)\\\\$", re.MULTILINE)
#: `value [lo, hi]`, with or without a \textbf wrapper.
VALUE_CI = re.compile(r"(\d+\.\d+)\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]")
NUMBER = re.compile(r"-?\d+\.\d+")
#: Any control sequence used in the manuscript.
MACRO_USE = re.compile(r"\\([a-zA-Z]+)")
#: A \newcommand definition in numbers.tex.
MACRO_DEF = re.compile(r"\\newcommand\{\\([a-zA-Z]+)\}")


def _load(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


paper_tables = _load("paper_tables")
paper_figures = _load("paper_figures")


@pytest.fixture(scope="module")
def snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    return paper_tables.load_results(RESULTS_DIR)


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Run both generators into a temporary tree, exactly as `make all` would."""
    root = tmp_path_factory.mktemp("paper1a")
    tables = root / "tables"
    figures = root / "figures"
    assert paper_tables.main(["--results-dir", str(RESULTS_DIR), "--out-dir", str(tables)]) == 0
    assert paper_figures.main(["--results-dir", str(RESULTS_DIR), "--out-dir", str(figures)]) == 0
    return tables, figures


def _unescape(arm: str) -> str:
    return arm.replace("\\_", "_")


def _rows(text: str) -> dict[str, str]:
    """Map an arm name to the rest of its row, for every row that starts with an arm."""
    return {_unescape(match.group("arm")): match.group("rest") for match in ROW.finditer(text)}


def test_every_table_is_written(generated: tuple[Path, Path]) -> None:
    tables, _ = generated
    missing = [name for name in EXPECTED_TABLES if not (tables / name).exists()]
    assert missing == [], f"generator did not write {missing}"


def test_every_figure_is_written(generated: tuple[Path, Path]) -> None:
    _, figures = generated
    missing = [name for name in EXPECTED_FIGURES if not (figures / name).exists()]
    assert missing == [], f"generator did not write {missing}"
    for name in EXPECTED_FIGURES:
        assert (figures / name).stat().st_size > 0


def test_parity_numbers_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    tables, _ = generated
    exp01, _ = snapshot
    rows = _rows((tables / "parity.tex").read_text())
    assert set(rows) == set(paper_tables.T0_ARMS)
    for arm, rest in rows.items():
        printed = NUMBER.findall(rest)
        assert len(printed) == 3, f"{arm}: expected three parity cells, got {printed}"
        expected = [
            f"{float(exp01['parity'][arm][key]['value']):.3f}"
            for key in ("eng_Latn", "hin_Deva", "eng_Latn__slp1")
        ]
        assert printed == expected, f"{arm}: {printed} != {expected}"


def test_controlled_numbers_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    tables, _ = generated
    _, exp02 = snapshot
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = {
        pair.split("/")[0]: pair for pair in paper_tables.controlled_pair_keys(exp02)
    }
    rows = _rows((tables / "tpp_controlled.tex").read_text())
    assert set(rows) == set(pairs), f"{set(rows)} != {set(pairs)}"
    for arm, rest in rows.items():
        printed = VALUE_CI.findall(rest)
        assert len(printed) == len(corpora), f"{arm}: got {printed}"
        for (value, low, high), corpus in zip(printed, corpora, strict=True):
            node = exp02["tpp_controlled"][corpus][pairs[arm]]
            assert value == f"{float(node['value']):.3f}", f"{arm}/{corpus} value"
            assert low == f"{float(node['ci_low']):.3f}", f"{arm}/{corpus} ci_low"
            assert high == f"{float(node['ci_high']):.3f}", f"{arm}/{corpus} ci_high"


def test_controlled_bolding_marks_only_intervals_below_one(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Bold means the whole interval sits below parity; nothing else may be bold."""
    tables, _ = generated
    _, exp02 = snapshot
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = {
        pair.split("/")[0]: pair for pair in paper_tables.controlled_pair_keys(exp02)
    }
    for arm, rest in _rows((tables / "tpp_controlled.tex").read_text()).items():
        cells = [cell for cell in rest.split("&") if VALUE_CI.search(cell)]
        for cell, corpus in zip(cells, corpora, strict=True):
            below = float(exp02["tpp_controlled"][corpus][pairs[arm]]["ci_high"]) < 1.0
            assert (r"\textbf" in cell) is below, f"{arm}/{corpus}: bolding disagrees"


def test_numbers_tex_defines_every_macro_the_manuscript_uses(
    generated: tuple[Path, Path],
) -> None:
    tables, _ = generated
    defined = set(MACRO_DEF.findall((tables / "numbers.tex").read_text()))

    missing_from_file = [name for name in paper_tables.MACROS if name not in defined]
    assert missing_from_file == [], f"MACROS names not emitted: {missing_from_file}"
    undeclared = [name for name in defined if name not in paper_tables.MACROS]
    assert undeclared == [], f"emitted but not listed in MACROS: {undeclared}"

    manuscript = (PAPER_DIR / "main.tex").read_text()
    used = {name for name in MACRO_USE.findall(manuscript) if name.startswith("num")}
    undefined = sorted(name for name in used if name not in defined)
    assert undefined == [], f"main.tex uses macros numbers.tex does not define: {undefined}"


def _macro_values(text: str) -> dict[str, str]:
    """Every `\\newcommand{\\name}{body}` in numbers.tex, as name -> body."""
    pattern = re.compile(r"\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}")
    return {name: body for name, body in pattern.findall(text)}


def test_renyi_macros_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """The macros §5.6 reads are the snapshot's numbers at the printed precision."""
    tables, _ = generated
    _, exp02 = snapshot
    values = _macro_values((tables / "numbers.tex").read_text())

    alpha = values["numRenyiAlpha"]
    assert alpha in {str(float(a)) for a in exp02["config"]["renyi_alphas"]}
    renyi = exp02["renyi"][paper_tables.RENYI_PROSE_CORPUS]
    english = exp02["renyi_english"][paper_tables.RENYI_PROSE_CORPUS]

    def sa(arm: str, variant: str) -> str:
        return f"{float(renyi[arm][variant][alpha]['value']):.3f}"

    band = [
        arm
        for arm in (*paper_tables.T0_ARMS, *paper_tables.T3_ARMS)
        if arm in renyi and arm != "T0_gpt2"
    ]
    assert len(band) >= 3, "the deployed band should not collapse to a couple of arms"
    for variant, lo_key, hi_key in (
        ("original", "numRenyiDeployedLo", "numRenyiDeployedHi"),
        ("slp1", "numRenyiDeployedSlpOneLo", "numRenyiDeployedSlpOneHi"),
    ):
        printed = sorted(sa(arm, variant) for arm in band)
        assert values[lo_key] == printed[0], lo_key
        assert values[hi_key] == printed[-1], hi_key

    assert values["numRenyiGptTwo"] == sa("T0_gpt2", "original")
    assert values["numRenyiGptTwoSlpOne"] == sa("T0_gpt2", "slp1")
    assert values["numRenyiBpeThirtyTwo"] == sa("T1_bpe_raw_32k", "slp1")
    assert values["numRenyiBpeSixtyFour"] == sa("T1_bpe_raw_64k", "slp1")
    unigram = sorted(sa(arm, "slp1") for arm in ("T2_unigram_raw_32k", "T2_unigram_raw_64k"))
    assert [values["numRenyiUnigramLo"], values["numRenyiUnigramHi"]] == unigram

    # The band is over the pivots and controls the paper reports, not every English arm
    # the snapshot happens to carry.
    en = sorted(
        f"{float(english[arm][alpha]['value']):.3f}"
        for arm in (*paper_tables.T0_ARMS, *paper_tables.controlled_english_arms())
        if arm in english
    )
    assert values["numRenyiEnglishLo"] == en[0]
    assert values["numRenyiEnglishHi"] == en[-1]


def test_renyi_prose_claims_hold_in_the_snapshot(
    snapshot: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The three orderings §5.6 asserts in words, checked against the numbers.

    The prose says the trained BPE arms sit above the deployed band, the trained Unigram
    arms below it, that the same ordering survives when the deployed arms are read in
    their SLP1 column rather than their original-script one, that the two Unigram arms sit
    at or just below the foot of the English band, and that R\u00e9nyi ranks the two BPE
    arms in the opposite order from the controlled TPP. If a re-run reverses any of those,
    the prose is wrong and this fails.
    """
    _, exp02 = snapshot
    alpha = paper_tables.RENYI_PROSE_ALPHA
    renyi = exp02["renyi"][paper_tables.RENYI_PROSE_CORPUS]
    english_arms = exp02["renyi_english"][paper_tables.RENYI_PROSE_CORPUS]
    deployed = [
        arm
        for arm in (*paper_tables.T0_ARMS, *paper_tables.T3_ARMS)
        if arm in renyi and arm != "T0_gpt2"
    ]
    band = [float(renyi[arm]["original"][alpha]["value"]) for arm in deployed]
    band_slp1 = [float(renyi[arm]["slp1"][alpha]["value"]) for arm in deployed]
    bpe = {
        arm: float(renyi[arm]["slp1"][alpha]["value"])
        for arm in ("T1_bpe_raw_32k", "T1_bpe_raw_64k")
    }
    unigram = [
        float(renyi[arm]["slp1"][alpha]["value"])
        for arm in ("T2_unigram_raw_32k", "T2_unigram_raw_64k")
    ]
    assert min(bpe.values()) > max(band), "BPE arms no longer sit above the deployed band"
    assert max(unigram) < min(band), "Unigram arms no longer sit below the deployed band"
    assert float(renyi["T0_gpt2"]["original"][alpha]["value"]) < min(band)

    # The trained arms have an SLP1 column only, so the sentence above compares across
    # columns. It states that the ordering survives the deployed arms' own SLP1 column.
    assert min(bpe.values()) > max(band_slp1), "BPE arms no longer clear the SLP1 band"
    assert max(unigram) < min(band_slp1), "Unigram arms no longer sit below the SLP1 band"

    # "At or just below the foot of" the English band: one Unigram arm inside it, one
    # under it, and neither above its top.
    english_band = [
        float(english_arms[arm][alpha]["value"])
        for arm in (*paper_tables.T0_ARMS, *paper_tables.controlled_english_arms())
        if arm in english_arms
    ]
    assert max(unigram) < max(english_band), "a Unigram arm now sits above the English band"
    assert min(unigram) < min(english_band), "no Unigram arm now sits below the English band"
    assert min(english_band) <= max(unigram), "both Unigram arms now sit below the band"

    controlled = exp02["tpp_controlled"]["samayik_test"]
    tpp_32 = float(controlled["T1_bpe_raw_32k/E1_bpe_32k"]["value"])
    tpp_64 = float(controlled["T1_bpe_raw_64k/E1_bpe_64k"]["value"])
    assert bpe["T1_bpe_raw_32k"] > bpe["T1_bpe_raw_64k"], "Renyi no longer prefers 32k"
    assert tpp_64 < tpp_32, "controlled TPP no longer prefers 64k"



#: The macros §5.5 reads, all of them written by `paper_tables.length_macros`. The
#: coverage test below fails if that function grows one this file does not check, so a
#: new number in the length prose cannot reach the manuscript unverified.
LENGTH_MACROS = (
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

#: The `$n$ pairs` row of a length table: counts, with thousands separators and a possible
#: sparse dagger.
N_ROW = re.compile(r"^\$n\$ pairs(?P<rest>.*?)\\\\$", re.MULTILINE)


def _dense(entries: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """A corpus x pair's populated, non-sparse bins, in bin order.

    Recomputed here rather than imported, so that the definition the prose rests on is
    stated twice and a change to either side shows up as a failure.
    """
    return [
        (name, node)
        for name, node in entries.items()
        if node.get("value") is not None and not node.get("sparse", False)
    ]


def _length_value_rows(text: str) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Every `value [lo, hi]` row of a length table, as (row label, cells), in order."""
    rows: list[tuple[str, list[tuple[str, str, str]]]] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if "&" not in stripped or not stripped.endswith(r"\\"):
            continue
        cells = VALUE_CI.findall(stripped)
        if cells:
            rows.append((stripped.split("&")[0].strip(), cells))
    return rows


def _length_count_rows(text: str) -> list[list[str]]:
    """Every `$n$ pairs` row of a length table, as its printed counts, in order."""
    return [re.findall(r"[\d,]+", match.group("rest")) for match in N_ROW.finditer(text)]


def _expected_length_rows(
    exp02: dict[str, Any], key: str, corpora: list[str], suppress_below: int | None = None
) -> tuple[list[tuple[str, list[tuple[str, str, str]]]], list[list[str]]]:
    """What one length table should print, from the JSON: its value rows and its `n` rows.

    `suppress_below` mirrors the body tables, which print the dagger alone for a bin
    holding fewer than that many pairs and leave the ratio to the appendix table.
    """
    pairs = paper_tables.controlled_pair_keys(exp02)
    bins = list(exp02[key][corpora[0]][pairs[0]].keys())
    values: list[tuple[str, list[tuple[str, str, str]]]] = []
    counts: list[list[str]] = []
    for corpus in corpora:
        entries = exp02[key][corpus]
        counts.append([f"{int(entries[pairs[0]][name]['n_pairs']):,}" for name in bins])
        for pair in pairs:
            label = paper_tables.matched_pair_short(pair.split("/")[0])
            cells = [
                (
                    f"{float(entries[pair][name]['value']):.3f}",
                    f"{float(entries[pair][name]['ci_low']):.3f}",
                    f"{float(entries[pair][name]['ci_high']):.3f}",
                )
                for name in bins
                if suppress_below is None
                or int(entries[pair][name]["n_pairs"]) >= suppress_below
            ]
            values.append((label, cells))
    return values, counts


def test_length_macro_coverage(snapshot: tuple[dict[str, Any], dict[str, Any]]) -> None:
    """Every macro the length section emits is one this file checks against the JSON."""
    _, exp02 = snapshot
    assert set(paper_tables.length_macros(exp02)) == set(LENGTH_MACROS)
    assert set(LENGTH_MACROS) <= set(paper_tables.MACROS)


def test_length_macros_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Every macro §5.5 reads, recomputed from `results.json` at the printed precision."""
    tables, _ = generated
    _, exp02 = snapshot
    values = _macro_values((tables / "numbers.tex").read_text())
    pairs = paper_tables.controlled_pair_keys(exp02)
    prose = paper_tables.LENGTH_PROSE_CORPUS
    verse = paper_tables.LENGTH_VERSE_CORPUS
    floor = int(exp02["config"]["length_sparse_below"])

    assert values["numLengthSparseBelow"] == str(floor)

    # The corpus means pool the English bins, weighted by the pairs in each.
    for prefix, corpus in (("numSamayik", prose), ("numItihasa", verse)):
        entries = exp02["tpp_by_length"][corpus][pairs[0]]
        total_pairs = sum(int(node["n_pairs"]) for node in entries.values())
        for suffix, field in (("MeanEnWords", "mean_words_en"), ("MeanSaWords", "mean_words_sa")):
            pooled = sum(
                int(node["n_pairs"]) * float(node[field]) for node in entries.values()
            )
            assert values[f"{prefix}{suffix}"] == f"{pooled / total_pairs:.1f}"

    flipped = 0
    total = 0
    for corpus in exp02["tpp_by_length"]:
        for pair in pairs:
            gradients = [
                float(_dense(exp02[key][corpus][pair])[-1][1]["value"])
                - float(_dense(exp02[key][corpus][pair])[0][1]["value"])
                for key in ("tpp_by_length", "tpp_by_length_sa")
            ]
            total += 1
            flipped += (gradients[0] > 0) != (gradients[1] > 0)
    assert values["numLengthGradientsFlipped"] == str(flipped)
    assert values["numLengthGradientsTotal"] == str(total)

    crossings = {
        "tpp_by_length": "numLengthPairsBelowOneEn",
        "tpp_by_length_sa": "numLengthPairsBelowOneSa",
    }
    for prefix, key in (("numLengthEn", "tpp_by_length"), ("numLengthSa", "tpp_by_length_sa")):
        names = [name for name, _node in _dense(exp02[key][prose][pairs[0]])]
        for position, name in (("First", names[0]), ("Last", names[-1])):
            cells = [float(exp02[key][prose][pair][name]["value"]) for pair in pairs]
            assert values[f"{prefix}{position}Bin"] == name.replace("-", "--")
            assert values[f"{prefix}{position}Lo"] == f"{min(cells):.3f}"
            assert values[f"{prefix}{position}Hi"] == f"{max(cells):.3f}"
        # The crossing the prose names: the last dense English bin, the first Sanskrit one.
        crossing = names[-1] if key == "tpp_by_length" else names[0]
        under = sum(1 for pair in pairs if float(exp02[key][prose][pair][crossing]["value"]) < 1.0)
        assert values[crossings[key]] == str(under)

    below = 0
    comparisons = 0
    gaps: list[float] = []
    for key in ("tpp_by_length", "tpp_by_length_sa"):
        for name in exp02[key][prose][pairs[0]]:
            if int(exp02[key][prose][pairs[0]][name]["n_pairs"]) < floor:
                continue
            if int(exp02[key][verse][pairs[0]][name]["n_pairs"]) < floor:
                continue
            for pair in pairs:
                comparisons += 1
                gap = float(exp02[key][prose][pair][name]["value"]) - float(
                    exp02[key][verse][pair][name]["value"]
                )
                gaps.append(gap)
                below += gap > 0
    assert values["numVerseBelowProseComparisons"] == str(below)
    assert values["numVerseBelowProseTotal"] == str(comparisons)
    assert values["numVerseProseGapMin"] == f"{min(gaps):.3f}"
    assert values["numVerseProseGapMax"] == f"{max(gaps):.3f}"

    # The illustrating bin is the jointly populated one holding the most verse pairs.
    for prefix, key, field, suffix in (
        ("numEnglishBin", "tpp_by_length", "mean_words_sa", "SaWords"),
        ("numSanskritBin", "tpp_by_length_sa", "mean_words_en", "EnWords"),
    ):
        joint = [
            name
            for name in exp02[key][prose][pairs[0]]
            if int(exp02[key][prose][pairs[0]][name]["n_pairs"]) >= floor
            and int(exp02[key][verse][pairs[0]][name]["n_pairs"]) >= floor
        ]
        busiest = max(joint, key=lambda name: int(exp02[key][verse][pairs[0]][name]["n_pairs"]))
        assert values[f"{prefix}Example"] == busiest.replace("-", "--")
        for label, corpus in ((f"{prefix}Itihasa", verse), (f"{prefix}Samayik", prose)):
            node = exp02[key][corpus][pairs[0]][busiest]
            assert values[f"{label}{suffix}"] == f"{float(node[field]):.1f}"


def test_length_prose_claims_hold_in_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """The two orderings §5.5 asserts in words, and every cell of the three length tables.

    §5.5 says that *every* within-corpus gradient changes sign between the two
    stratifications, and that verse sits below prose at *every* jointly populated bin
    under both. Neither is a number the prose could soften: if a re-run breaks either, the
    subsection is wrong and this fails rather than the macro quietly changing.
    """
    tables, _ = generated
    _, exp02 = snapshot

    flipped, gradients = paper_tables.length_gradient_flips(exp02)
    assert gradients > 0
    assert flipped == gradients, "a gradient keeps its sign under both stratifications"

    below, comparisons, gap_min, gap_max = paper_tables.verse_below_prose(exp02)
    assert comparisons > 0
    assert below == comparisons, "verse is not below prose at every jointly populated bin"
    assert 0 < gap_min <= gap_max

    body = list(paper_tables.LENGTH_BODY_CORPORA)
    every = list(exp02["tpp_by_length"].keys())
    floor = paper_tables.LENGTH_BODY_SUPPRESS_BELOW
    for name, blocks, suppress in (
        (
            "tpp_by_length.tex",
            (("tpp_by_length", body), ("tpp_by_length_sa", body)),
            floor,
        ),
        (
            "tpp_by_length_all.tex",
            (("tpp_by_length", every), ("tpp_by_length_sa", every)),
            None,
        ),
    ):
        text = (tables / name).read_text()
        expected_values: list[tuple[str, list[tuple[str, str, str]]]] = []
        expected_counts: list[list[str]] = []
        for key, corpora in blocks:
            values, counts = _expected_length_rows(exp02, key, corpora, suppress)
            expected_values.extend(values)
            expected_counts.extend(counts)
        assert _length_value_rows(text) == expected_values, f"{name}: value cells"
        assert _length_count_rows(text) == expected_counts, f"{name}: $n$ rows"


def test_thin_length_bins_are_suppressed_in_the_body_but_kept_in_the_appendix(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """A bin under the body floor prints the dagger alone; the appendix still carries it."""
    tables, _ = generated
    _, exp02 = snapshot
    pairs = paper_tables.controlled_pair_keys(exp02)
    floor = paper_tables.LENGTH_BODY_SUPPRESS_BELOW
    thin = [
        (key, corpus, name)
        for key in ("tpp_by_length", "tpp_by_length_sa")
        for corpus in paper_tables.LENGTH_BODY_CORPORA
        for name, node in exp02[key][corpus][pairs[0]].items()
        if int(node["n_pairs"]) < floor
    ]
    assert thin, "no bin in the body corpora is thin enough to exercise the suppression"
    appendix = (tables / "tpp_by_length_all.tex").read_text()
    body = (tables / "tpp_by_length.tex").read_text()
    for key, corpus, name in thin:
        for pair in pairs:
            printed = f"{float(exp02[key][corpus][pair][name]['value']):.3f}"
            assert printed not in body, f"the body table still prints the {name} bin"
            assert printed in appendix, f"the appendix lost the {name} bin"


def test_no_bolding_in_the_length_tables(generated: tuple[Path, Path]) -> None:
    """Bold in a table whose caption calls the gradient an artefact sends two signals."""
    tables, _ = generated
    for name in ("tpp_by_length.tex", "tpp_by_length_all.tex"):
        # Only the tabular: the caption bolds the name of the side the bins are cut on.
        for block in (tables / name).read_text().split(r"\begin{tabular}")[1:]:
            body = block.split(r"\end{tabular}")[0]
            assert r"\textbf" not in body, f"{name} still bolds cells"


def test_hindi_table_drops_the_slp1_artifact_column(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """The column its own caption said not to read is no longer printed."""
    tables, _ = generated
    _, exp02 = snapshot
    text = (tables / "tpp_hindi.tex").read_text()
    assert "SLP1)" not in text.split(r"\caption")[0], "the SLP1 column is still in the body"
    for arm, node in exp02["tpp_hindi"].items():
        assert f"{float(node['original']['value']):.3f}" in text, arm
        assert f"{float(node['slp1']['value']):.3f}" not in text, arm


def test_controlled_pairs_are_restricted_to_the_reported_four(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """A pair the snapshot gains without prose to go with it must not reach a table."""
    tables, _ = generated
    _, exp02 = snapshot
    assert paper_tables.controlled_pair_keys(exp02) == [
        pair
        for pair in paper_tables.CONTROLLED_PAIRS
        if pair in {f"{a}/{b}" for a, b in exp02["config"]["controlled_pairs"]}
    ]
    rows = _rows((tables / "tpp_controlled.tex").read_text())
    assert set(rows) == {pair.split("/")[0] for pair in paper_tables.controlled_pair_keys(exp02)}


def test_preregistration_marks_the_untested_arm_as_untested(
    generated: tuple[Path, Path],
) -> None:
    """An Outcome verdict on a prediction for an arm this paper does not build is wrong."""
    text = (generated[0] / "preregistration.tex").read_text()
    assert "Untested: the predicted arm is not built here" in text
    assert "Observed against the pivot" not in text


def test_training_split_sizes_agree_with_the_data_readme() -> None:
    """The two constants taken from `data/README.md` still match that file."""
    readme = (REPO_ROOT / "data" / "README.md").read_text()
    assert f"train {paper_tables.SAMAYIK_TRAIN_PAIRS:,} / dev 2,416" in readme
    assert f"train {paper_tables.ITIHASA_TRAIN_PAIRS:,} / dev 6,148" in readme


def test_committed_tables_are_current(
    snapshot: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """What is checked in under `paper/1a/tables` is what the generator produces now."""
    exp01, exp02 = snapshot
    for name, content in paper_tables.build(exp01, exp02).items():
        committed = PAPER_DIR / "tables" / name
        assert committed.exists(), f"{name} has not been generated into paper/1a/tables"
        assert committed.read_text() == content, f"{name} is stale; re-run paper_tables.py"


def test_absent_analyses_degrade_to_a_comment() -> None:
    """The two optional writers stay usable against a snapshot that lacks their keys."""
    stripped: dict[str, Any] = {key: {} for key in ("config",)}
    assert paper_tables.tpp_by_length_table(stripped).startswith("%")
    assert paper_tables.tpp_by_length_all_tables(stripped).startswith("%")
    assert paper_tables.renyi_table(stripped).startswith("%")
    assert paper_figures.figure_length(stripped) is None


def test_no_forbidden_framing_in_the_manuscript() -> None:
    """CLAUDE.md forbids the NASA claim outright and perplexity as a cross-arm metric."""
    text = (PAPER_DIR / "main.tex").read_text().lower()
    assert "nasa" not in text
    assert "perplexity" not in text.replace(
        "we do not report perplexity anywhere", ""
    ) or "not comparable across tokenizers" in text


def test_snapshot_is_never_written(generated: tuple[Path, Path]) -> None:
    """The generators read `results/` and write only where they are told to."""
    tables, figures = generated
    for path in (tables, figures):
        assert RESULTS_DIR not in path.parents and path != RESULTS_DIR
    before = json.loads((RESULTS_DIR / "01_baseline_penalty" / "results.json").read_text())
    assert "parity" in before
