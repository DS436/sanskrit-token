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
    pairs = {pair.split("/")[0]: pair for pair in exp02["tpp_controlled"][corpora[0]]}
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
    pairs = {pair.split("/")[0]: pair for pair in exp02["tpp_controlled"][corpora[0]]}
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

    en = sorted(f"{float(node[alpha]['value']):.3f}" for node in english.values())
    assert values["numRenyiEnglishLo"] == en[0]
    assert values["numRenyiEnglishHi"] == en[-1]


def test_renyi_prose_claims_hold_in_the_snapshot(
    snapshot: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The three orderings §5.6 asserts in words, checked against the numbers.

    The prose says the trained BPE arms sit above the deployed band, the trained Unigram
    arms below it, and that R\u00e9nyi ranks the two BPE arms in the opposite order from the
    controlled TPP. If a re-run reverses any of those, the prose is wrong and this fails.
    """
    _, exp02 = snapshot
    alpha = paper_tables.RENYI_PROSE_ALPHA
    renyi = exp02["renyi"][paper_tables.RENYI_PROSE_CORPUS]
    band = [
        float(renyi[arm]["original"][alpha]["value"])
        for arm in (*paper_tables.T0_ARMS, *paper_tables.T3_ARMS)
        if arm in renyi and arm != "T0_gpt2"
    ]
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

    controlled = exp02["tpp_controlled"]["samayik_test"]
    tpp_32 = float(controlled["T1_bpe_raw_32k/E1_bpe_32k"]["value"])
    tpp_64 = float(controlled["T1_bpe_raw_64k/E1_bpe_64k"]["value"])
    assert bpe["T1_bpe_raw_32k"] > bpe["T1_bpe_raw_64k"], "Renyi no longer prefers 32k"
    assert tpp_64 < tpp_32, "controlled TPP no longer prefers 64k"


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
    stripped = {key: {} for key in ("config",)}
    assert paper_tables.tpp_by_length_table(stripped).startswith("%")
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
