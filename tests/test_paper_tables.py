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
import statistics
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
    "tpp_deployed_all.tex",
    "tpp_controlled.tex",
    "decomposition.tex",
    "block_ci.tex",
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
    "vocab.pdf",
    "vocab.png",
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


def _controlled_rows(text: str, pairs: list[str]) -> dict[str, list[str]]:
    """Map each reported pair to its row's cells, in the table's own row order.

    The controlled table labels a row by the pair's short name and its control, not by the
    arm, and carries two rows per Sanskrit arm, so the rows are matched to
    `CONTROLLED_PAIRS` positionally after the header.
    """
    body = [
        line
        for line in text.splitlines()
        if line.endswith(r"\\") and (" & " in line) and "Matched pair" not in line
    ]
    assert len(body) == len(pairs), f"{len(body)} rows for {len(pairs)} pairs"
    return {pair: row.rstrip("\\").split("&") for pair, row in zip(pairs, body, strict=True)}


def test_controlled_numbers_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    tables, _ = generated
    _, exp02 = snapshot
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = paper_tables.reported_pairs(exp02)
    rows = _controlled_rows((tables / "tpp_controlled.tex").read_text(), pairs)
    for pair, cells in rows.items():
        sa_arm, en_arm = pair.split("/")
        assert paper_tables.matched_pair_short(sa_arm) == cells[0].strip()
        expected_control = "E1_bm" if en_arm.endswith("_bm") else "E1"
        assert expected_control in _unescape(cells[1]), f"{pair}: control column"
        printed = VALUE_CI.findall(" & ".join(cells[2:]))
        assert len(printed) == len(corpora), f"{pair}: got {printed}"
        for (value, low, high), corpus in zip(printed, corpora, strict=True):
            node = exp02["tpp_controlled"][corpus][pair]
            assert value == f"{float(node['value']):.3f}", f"{pair}/{corpus} value"
            assert low == f"{float(node['ci_low']):.3f}", f"{pair}/{corpus} ci_low"
            assert high == f"{float(node['ci_high']):.3f}", f"{pair}/{corpus} ci_high"


def test_controlled_bolding_marks_only_intervals_below_one(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Bold means the whole interval sits below parity; nothing else may be bold."""
    tables, _ = generated
    _, exp02 = snapshot
    corpora = list(exp02["tpp_controlled"].keys())
    pairs = paper_tables.reported_pairs(exp02)
    rows = _controlled_rows((tables / "tpp_controlled.tex").read_text(), pairs)
    for pair, cells in rows.items():
        values = [cell for cell in cells[2:] if VALUE_CI.search(cell)]
        for cell, corpus in zip(values, corpora, strict=True):
            below = float(exp02["tpp_controlled"][corpus][pair]["ci_high"]) < 1.0
            assert (r"\textbf" in cell) is below, f"{pair}/{corpus}: bolding disagrees"


def test_undersized_controls_are_marked_in_the_controlled_table(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """A control the trainer could not bring to its requested size carries a dagger."""
    tables, _ = generated
    _, exp02 = snapshot
    pairs = paper_tables.reported_pairs(exp02)
    rows = _controlled_rows((tables / "tpp_controlled.tex").read_text(), pairs)
    undersized = set(paper_tables.undersized_control_arms(exp02))
    assert undersized, "the snapshot no longer carries an undersized control arm"
    for pair, cells in rows.items():
        marked = "ddagger" in cells[1]
        assert marked is (pair.split("/")[1] in undersized), f"{pair}: dagger disagrees"
    caption = (tables / "tpp_controlled.tex").read_text()
    for arm in undersized:
        assert f"{paper_tables.vocab_size(exp02, arm):,}" in caption


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
        for arm in (*paper_tables.T0_ARMS, *paper_tables.RENYI_ENGLISH_ARMS)
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
        for arm in (*paper_tables.T0_ARMS, *paper_tables.RENYI_ENGLISH_ARMS)
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


def test_reported_pairs_are_restricted_to_the_ones_the_prose_discusses(
    snapshot: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """A pair the snapshot gains without prose to go with it must not reach a table."""
    _, exp02 = snapshot
    configured = {f"{a}/{b}" for a, b in exp02["config"]["controlled_pairs"]}
    assert paper_tables.reported_pairs(exp02) == [
        pair for pair in paper_tables.CONTROLLED_PAIRS if pair in configured
    ]
    # The byte reference is a pair in the snapshot and is never a matched pair here.
    assert paper_tables.BYTE_PAIR not in paper_tables.CONTROLLED_PAIRS
    # The length strata skip the 128k arms, the byte-matched controls and the reference,
    # so the length blocks run over the four pairs the snapshot actually stratifies.
    assert paper_tables.controlled_pair_keys(exp02) == list(paper_tables.LENGTH_PAIRS)
    for pair in paper_tables.LENGTH_PAIRS:
        assert pair in exp02["tpp_by_length"]["samayik_test"]


def test_preregistration_scopes_the_crossing_to_where_it_was_observed(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """The fifth row's Outcome names both the pivot and the one controlled crossing."""
    _, exp02 = snapshot
    text = (generated[0] / "preregistration.tex").read_text()
    assert "\\texttt{T6} is not built here" in text
    assert "Observed against the pivot; under the matched control only at" in text
    huge = f"{paper_tables.requested_vocab('T1_bpe_raw_128k'):,} in domain"
    assert huge in text
    for corpus, pair in (
        ("samayik_test", "T1_bpe_raw_128k/E1_bpe_128k"),
        ("samayik_test_ood", "T1_bpe_raw_128k/E1_bpe_128k"),
    ):
        node = exp02["tpp_controlled"][corpus][pair]
        assert f"{float(node['value']):.3f}" in text


def test_training_split_sizes_agree_with_the_data_readme() -> None:
    """The two constants taken from `data/README.md` still match that file."""
    readme = (REPO_ROOT / "data" / "README.md").read_text()
    assert f"train {paper_tables.SAMAYIK_TRAIN_PAIRS:,} / dev 2,416" in readme
    assert f"train {paper_tables.ITIHASA_TRAIN_PAIRS:,} / dev 6,148" in readme


def test_training_corpus_bytes_agree_with_the_experiment_readme() -> None:
    """The byte constants §5.3 and the Limitations quote still match Experiment 02.

    They are constants because the training corpora live under the gitignored
    `data/processed/` and the tracked snapshot records no corpus size, so the generator
    cannot read them. This is the check that keeps them honest.
    """
    readme = " ".join(
        (REPO_ROOT / "experiments" / "02_tpp_parallel" / "README.md").read_text().split()
    )
    for value in (
        paper_tables.SANSKRIT_TRAIN_BYTES,
        paper_tables.ENGLISH_TRAIN_BYTES,
        paper_tables.ENGLISH_BM_TRAIN_BYTES,
    ):
        assert f"{value:,} bytes" in readme, f"{value:,} bytes not in the README"
    assert f"{paper_tables.ENGLISH_BM_TRAIN_LINES:,} lines" in readme
    assert f"{paper_tables.ENGLISH_TRAIN_LINES:,} training sentences" in readme


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


# --------------------------------------------------------------------------------------
# The 2026-09-08 wave: byte-matched control, vocabulary sweep, decomposition, block CIs
# --------------------------------------------------------------------------------------


def _byte_matched_moves(exp02: dict[str, Any]) -> list[float]:
    """Every pair-matched ratio's move when the control is byte-matched instead."""
    controlled = exp02["tpp_controlled"]
    moves: list[float] = []
    for corpus in controlled:
        for pair in paper_tables.reported_pairs(exp02):
            sa_arm, en_arm = pair.split("/")
            if en_arm.endswith("_bm"):
                continue
            twin = f"{sa_arm}/{en_arm}_bm"
            moves.append(
                float(controlled[corpus][twin]["value"])
                - float(controlled[corpus][pair]["value"])
            )
    return moves


def test_byte_matched_macros_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """§5.3's byte-matched paragraph, recomputed from the JSON and the two constants."""
    tables, _ = generated
    _, exp02 = snapshot
    values = _macro_values((tables / "numbers.tex").read_text())
    moves = _byte_matched_moves(exp02)
    sizes = sorted(abs(move) for move in moves)

    assert values["numBmMoveCount"] == str(len(moves))
    assert values["numBmMaxMove"] == f"{max(sizes):.3f}"
    assert values["numBmMedianMove"] == f"{statistics.median(sizes):.3f}"
    assert values["numBmDownwardCount"] == str(sum(1 for move in moves if move < 0))
    assert values["numBmLines"] == f"{paper_tables.ENGLISH_BM_TRAIN_LINES:,}"
    assert values["numBmLinePct"] == (
        f"{paper_tables.ENGLISH_BM_TRAIN_LINES / paper_tables.ENGLISH_TRAIN_LINES * 100:.1f}"
    )
    for name, constant in (
        ("numBytesSanskritTrain", paper_tables.SANSKRIT_TRAIN_BYTES),
        ("numBytesEnglishTrain", paper_tables.ENGLISH_TRAIN_BYTES),
        ("numBytesEnglishBmTrain", paper_tables.ENGLISH_BM_TRAIN_BYTES),
    ):
        assert values[name] == f"{constant / 1_000_000:.2f}"
    excess = paper_tables.ENGLISH_TRAIN_BYTES / paper_tables.SANSKRIT_TRAIN_BYTES - 1.0
    assert values["numBytesEnglishExcessPct"] == f"{excess * 100:.0f}"


def test_vocabulary_macros_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """The 128k numbers §5.3 and the abstract read."""
    tables, _ = generated
    _, exp02 = snapshot
    values = _macro_values((tables / "numbers.tex").read_text())
    controlled = exp02["tpp_controlled"]
    for key, corpus, pair in (
        ("numTppBpeHugeSamayik", "samayik_test", "T1_bpe_raw_128k/E1_bpe_128k"),
        ("numTppBpeHugeSamayikBm", "samayik_test", "T1_bpe_raw_128k/E1_bpe_128k_bm"),
        ("numTppBpeHugeOod", "samayik_test_ood", "T1_bpe_raw_128k/E1_bpe_128k"),
    ):
        node = controlled[corpus][pair]
        assert values[key] == f"{float(node['value']):.3f}"
        assert values[f"{key}Ci"] == (
            f"[{float(node['ci_low']):.3f}, {float(node['ci_high']):.3f}]"
        )
    ood = controlled["samayik_test_ood"]["T1_bpe_raw_128k/E1_bpe_128k"]
    assert values["numTppBpeHugeOodBlockCi"] == (
        f"[{float(ood['ci_low_block']):.3f}, {float(ood['ci_high_block']):.3f}]"
    )
    flores = controlled["flores_devtest"]["T1_bpe_raw_128k/E1_bpe_128k"]
    assert values["numTppBpeHugeFlores"] == f"{float(flores['value']):.3f}"
    assert values["numVocabHuge"] == f"{paper_tables.requested_vocab('T1_bpe_raw_128k'):,}"
    assert values["numUnigramBmPieces"] == (
        f"{paper_tables.vocab_size(exp02, 'E1_unigram_64k_bm'):,}"
    )
    assert values["numUnigramHugePieces"] == (
        f"{paper_tables.vocab_size(exp02, 'E1_unigram_128k'):,}"
    )
    small = paper_tables.reported_pairs(exp02, paper_tables.SMALL_VOCAB_TOKENS)
    huge = paper_tables.reported_pairs(exp02, (paper_tables.HUGE_VOCAB_TOKEN,))
    assert values["numControlledPairsSmall"] == str(len(small)) == "8"
    assert values["numControlledPairsHuge"] == str(len(huge)) == "4"
    assert values["numControlledPairsAll"] == str(len(small) + len(huge))
    out_of_domain = [
        float(controlled[corpus][pair]["value"])
        for corpus in ("samayik_test_ood", "flores_devtest")
        for pair in huge
    ]
    assert values["numTppControlledHugeOodLo"] == f"{min(out_of_domain):.3f}"
    assert values["numTppControlledHugeOodHi"] == f"{max(out_of_domain):.3f}"


def test_decomposition_macros_and_table_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """§5.4's numbers, and the identity the whole section rests on."""
    tables, _ = generated
    _, exp02 = snapshot
    values = _macro_values((tables / "numbers.tex").read_text())
    prose = exp02["side_decomposition"]["samayik_test"]
    verse = exp02["side_decomposition"]["itihasa_test"]
    small = paper_tables.reported_pairs(exp02, paper_tables.SMALL_VOCAB_TOKENS)
    huge = paper_tables.reported_pairs(exp02, (paper_tables.HUGE_VOCAB_TOKEN,))

    # The factorisation is exact, which is what lets the section attribute the ratio.
    for node in (*prose.values(), *verse.values()):
        product = float(node["char_ratio"]) * float(node["density_ratio"])
        assert abs(product - float(node["tpp"])) < 1e-9

    char_prose = float(prose[small[0]]["char_ratio"])
    char_verse = float(verse[small[0]]["char_ratio"])
    assert values["numCharRatioSamayik"] == f"{char_prose:.3f}"
    assert values["numCharRatioItihasa"] == f"{char_verse:.3f}"
    assert values["numCharRatioFactor"] == f"{char_prose / char_verse:.3f}"
    huge_bpe = [pair for pair in huge if "_bpe_" in pair.split("/")[0]]
    for key, node, pairs in (
        ("numDensityRatioSamayik", prose, small),
        ("numDensityRatioItihasa", verse, small),
        ("numDensityRatioSamayikHuge", prose, huge_bpe),
    ):
        band = [float(node[pair]["density_ratio"]) for pair in pairs]
        assert values[f"{key}Lo"] == f"{min(band):.3f}"
        assert values[f"{key}Hi"] == f"{max(band):.3f}"
    gap = max(
        abs(float(prose[pair]["density_ratio"]) - float(verse[pair]["density_ratio"]))
        for pair in small
    )
    assert values["numDensityPairGap"] == f"{gap:.3f}"
    assert values["numTSevenSamayik"] == f"{float(prose[paper_tables.BYTE_PAIR]['tpp']):.3f}"
    assert values["numTSevenItihasa"] == f"{float(verse[paper_tables.BYTE_PAIR]['tpp']):.3f}"
    assert values["numTSevenVocab"] == f"{paper_tables.vocab_size(exp02, 'T7_byt5'):,}"

    # Every printed cell of the table is in the snapshot at the printed precision.
    text = (tables / "decomposition.tex").read_text()
    for pair in (*small, *[p for p in huge if "_bpe_" in p and not p.endswith("_bm")]):
        for node in (prose[pair], verse[pair]):
            for key in ("char_ratio", "density_ratio", "tpp"):
                assert f"{float(node[key]):.3f}" in text, f"{pair} {key}"


def test_block_macros_match_the_snapshot(
    generated: tuple[Path, Path], snapshot: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """The widths Appendix \\ref{sec:blockci} states, recomputed."""
    tables, _ = generated
    _, exp02 = snapshot
    values = _macro_values((tables / "numbers.tex").read_text())
    controlled = exp02["tpp_controlled"]
    pairs = [*paper_tables.reported_pairs(exp02), paper_tables.BYTE_PAIR]
    assert values["numBlockRowsPerCorpus"] == str(len(pairs))
    assert values["numBlockLength"] == str(
        int(controlled["samayik_test"][pairs[0]]["block_length"])
    )
    for key, corpus in (
        ("numBlockWidenSamayik", "samayik_test"),
        ("numBlockWidenOod", "samayik_test_ood"),
        ("numBlockWidenItihasa", "itihasa_test"),
        ("numBlockWidenFlores", "flores_devtest"),
    ):
        ratios = [
            (
                float(controlled[corpus][pair]["ci_high_block"])
                - float(controlled[corpus][pair]["ci_low_block"])
            )
            / (
                float(controlled[corpus][pair]["ci_high"])
                - float(controlled[corpus][pair]["ci_low"])
            )
            for pair in pairs
        ]
        assert values[f"{key}Lo"] == f"{min(ratios):.1f}"
        assert values[f"{key}Hi"] == f"{max(ratios):.1f}"
    assert values["numBlockItihasaMaxUpper"] == (
        f"{max(float(controlled['itihasa_test'][p]['ci_high_block']) for p in pairs):.3f}"
    )
    text = (tables / "block_ci.tex").read_text()
    for corpus in controlled:
        for pair in pairs:
            node = controlled[corpus][pair]
            assert (
                f"[{float(node['ci_low_block']):.3f}, {float(node['ci_high_block']):.3f}]"
                in text
            ), f"{corpus}/{pair} block interval missing from the table"


def test_wave_one_prose_claims_hold_in_the_snapshot(
    snapshot: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The ordinal claims of the abstract, §5.3 and §5.4, checked against the numbers.

    Every one of these is a sentence in the manuscript rather than a printed value, so a
    re-run that reverses one leaves the prose wrong and nothing else would catch it.
    """
    _, exp02 = snapshot
    controlled = exp02["tpp_controlled"]

    def verdict(node: dict[str, Any]) -> str:
        if float(node["ci_low"]) > 1.0:
            return "above"
        return "below" if float(node["ci_high"]) < 1.0 else "straddles"

    # (1) The byte-matched control changes no verdict, at any size.
    for corpus in controlled:
        for pair in paper_tables.reported_pairs(exp02):
            sa_arm, en_arm = pair.split("/")
            if en_arm.endswith("_bm"):
                continue
            twin = controlled[corpus][f"{sa_arm}/{en_arm}_bm"]
            assert verdict(controlled[corpus][pair]) == verdict(twin), f"{corpus}/{pair}"

    # (1) At 32k and 64k every prose and FLORES pair is above parity under both controls,
    # and every verse pair is below it.
    for pair in paper_tables.reported_pairs(exp02, paper_tables.SMALL_VOCAB_TOKENS):
        for corpus in ("samayik_test", "samayik_test_ood", "flores_devtest"):
            assert float(controlled[corpus][pair]["ci_low"]) > 1.0, f"{corpus}/{pair}"
        assert float(controlled["itihasa_test"][pair]["ci_high"]) < 1.0, pair

    # (2) The BPE ratio falls with the vocabulary size on every corpus and both controls,
    # strictly at every step except in one sequence, which the prose names.
    series = paper_tables.bpe_progression(exp02)
    assert len(series) == 2 * len(controlled)
    monotone = 0
    for name, values in series.items():
        assert values[2] < values[0] and values[2] < values[1], f"{name} not falling"
        if values[0] > values[1] > values[2]:
            monotone += 1
    assert monotone == len(series) - 1
    exception = [name for name, v in series.items() if not v[0] > v[1] > v[2]]
    assert exception == ["flores_devtest/E1"], exception

    # (2) At 128k, in-domain prose crosses under both controls and stays above parity out
    # of domain and on FLORES.
    for suffix in ("", "_bm"):
        pair = f"T1_bpe_raw_128k/E1_bpe_128k{suffix}"
        assert float(controlled["samayik_test"][pair]["ci_high"]) < 1.0
        for corpus in ("samayik_test_ood", "flores_devtest"):
            assert float(controlled[corpus][pair]["ci_low"]) > 1.0, corpus

    # (3) The density ratios sit in a narrow band about 1 and agree between the corpora,
    # so the verse crossing lives in the character ratio.
    prose = exp02["side_decomposition"]["samayik_test"]
    verse = exp02["side_decomposition"]["itihasa_test"]
    small = paper_tables.reported_pairs(exp02, paper_tables.SMALL_VOCAB_TOKENS)
    for pair in small:
        for node in (prose[pair], verse[pair]):
            assert 0.99 < float(node["density_ratio"]) < 1.12, pair
        gap = abs(float(prose[pair]["density_ratio"]) - float(verse[pair]["density_ratio"]))
        assert gap <= 0.032, f"{pair}: density ratios diverge by {gap:.3f}"
    assert float(prose[small[0]]["char_ratio"]) > 1.0, "prose Sanskrit is no longer longer"
    assert float(verse[small[0]]["char_ratio"]) < 1.0, "verse Sanskrit is no longer shorter"

    # (3) The byte reference is the two sides' byte ratio and nothing else.
    for corpus, node in exp02["side_decomposition"].items():
        entry = node[paper_tables.BYTE_PAIR]
        expected = float(entry["bytes_sa"]) / float(entry["bytes_en"])
        assert abs(float(entry["tpp"]) - expected) < 1e-9, corpus

    # (4) Blocking never narrows a verse interval, and every verse verdict survives it.
    for pair in (*paper_tables.reported_pairs(exp02), paper_tables.BYTE_PAIR):
        node = controlled["itihasa_test"][pair]
        iid = float(node["ci_high"]) - float(node["ci_low"])
        block = float(node["ci_high_block"]) - float(node["ci_low_block"])
        assert block >= iid, f"{pair}: block interval is narrower on verse"
        assert float(node["ci_high_block"]) < 1.0, pair

    # (4) The one controlled verdict the block bootstrap changes is the 128k BPE pair on
    # the out-of-domain prose split, under either control.
    changed = [
        (corpus, pair)
        for corpus in controlled
        for pair in paper_tables.reported_pairs(exp02)
        if verdict(controlled[corpus][pair])
        != (
            "above"
            if float(controlled[corpus][pair]["ci_low_block"]) > 1.0
            else "below"
            if float(controlled[corpus][pair]["ci_high_block"]) < 1.0
            else "straddles"
        )
    ]
    assert changed == [
        ("samayik_test_ood", "T1_bpe_raw_128k/E1_bpe_128k"),
        ("samayik_test_ood", "T1_bpe_raw_128k/E1_bpe_128k_bm"),
    ], changed
