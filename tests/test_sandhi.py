"""Tests for the sandhi-splitting wrapper and its on-disk cache.

Everything here is offline: `SandhiSplitter` loads `torch`/`transformers` lazily, inside
the loader, so a splitter constructed with `model_id=None` and a monkeypatched
`_generate_iast` never touches the network, the 2.3 GB checkpoint, or `torch` at all.
That seam is deliberately narrow — `_generate_iast` receives already-prefixed IAST
strings and returns decoded IAST — so prefixing, chunking, batch ordering, the cache and
both transliteration directions stay real code under test (plan Task 2, Step 1).

The one test that loads the real checkpoint is gated on `SANSKRIT_TOK_NETWORK_TESTS`, the
convention `tests/test_registry.py` and `tests/test_flores.py` already use.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from sanskrit_tok.encoding import from_slp1, to_slp1
from sanskrit_tok.sandhi import SandhiSplitter
from sanskrit_tok.sandhi.byt5 import (
    CHUNK_MAX_BYTES,
    MAX_BYTES,
    SEGMENT_SEPARATOR,
    SEGMENTATION_PREFIX,
    SPLITTER_CANDIDATES,
    chunk_text,
)
from sanskrit_tok.sandhi.cache import SplitCache

# A real Sāmayik-style Devanagari sentence and the IAST the model would see for it.
DEVA = "तत् अपि गच्छति"
LONG_DEVA = "रामः वनम् गच्छति ।"


def _iast(text: str) -> str:
    """The IAST form the splitter feeds the model, via SLP1 (the pipeline's own path)."""
    return from_slp1(to_slp1(text, "devanagari"), "iast")


# --- SplitCache -------------------------------------------------------------------


def test_cache_roundtrips_through_a_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = SplitCache(path)
    assert cache.get(DEVA) is None
    cache.put(DEVA, "tat api gacCati")
    cache.flush()

    assert cache.get(DEVA) == "tat api gacCati"
    assert len(cache) == 1
    reopened = SplitCache(path)
    assert reopened.get(DEVA) == "tat api gacCati"
    assert len(reopened) == 1


def test_cache_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = SplitCache(path)
    cache.put(DEVA, "tat api gacCati")
    cache.flush()

    (line,) = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert set(record) == {"key", "input", "output"}
    assert record["input"] == DEVA
    assert record["output"] == "tat api gacCati"
    assert record["key"] == SplitCache.key(DEVA)


def test_cache_key_ignores_surrounding_whitespace(tmp_path: Path) -> None:
    cache = SplitCache(tmp_path / "cache.jsonl")
    cache.put(f"  {DEVA}\n", "tat api gacCati")
    assert cache.get(DEVA) == "tat api gacCati"
    assert SplitCache.key(DEVA) == SplitCache.key(f"\t{DEVA} ")


def test_cache_drops_a_truncated_last_line_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "cache.jsonl"
    good = json.dumps({"key": SplitCache.key(DEVA), "input": DEVA, "output": "tat api"})
    path.write_text(good + '\n{"key": "abc", "inp', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        cache = SplitCache(path)

    assert len(cache) == 1
    assert cache.get(DEVA) == "tat api"
    assert any("truncated" in record.message.lower() for record in caplog.records)


def test_cache_appends_rather_than_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    cache = SplitCache(path)
    cache.put("a", "a")
    cache.flush()
    cache.put("b", "b")
    cache.flush()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(SplitCache(path)) == 2


def test_cache_on_a_missing_file_starts_empty(tmp_path: Path) -> None:
    cache = SplitCache(tmp_path / "nested" / "cache.jsonl")
    assert len(cache) == 0
    assert cache.get(DEVA) is None


# --- SandhiSplitter ---------------------------------------------------------------


class FakeGenerator:
    """Stand-in for `_generate_iast`: echoes each prefixed input as two IAST segments.

    It records every batch it was handed, so a test can assert on batch sizes and on the
    order the splitter fed the model in (which length sorting deliberately changes).
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, batch: list[str]) -> list[str]:
        self.batches.append(list(batch))
        outputs = []
        for prefixed in batch:
            assert prefixed.startswith(SEGMENTATION_PREFIX)
            outputs.append(prefixed[len(SEGMENTATION_PREFIX) :] + " iti")
        return outputs

    @property
    def seen(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


def _splitter(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> tuple[
    SandhiSplitter, FakeGenerator
]:
    splitter = SandhiSplitter(**kwargs)  # type: ignore[arg-type]
    fake = FakeGenerator()
    monkeypatch.setattr(splitter, "_generate_iast", fake)
    return splitter, fake


def test_split_turns_the_models_underscore_separator_into_a_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real checkpoint separates segments with `_`; one whitespace unit per segment.

    Verified against `chronbmm/sanskrit5-multitask`, whose raw decode of
    `S viśvāsajanakam ātmānam` is `viśvāsa_janakam_ātmānam_` — trailing separator and all.
    """
    splitter = SandhiSplitter()
    monkeypatch.setattr(
        splitter,
        "_generate_iast",
        lambda batch: ["viśvāsa_janakam_ātmānam_" for _ in batch],
    )
    (output,) = splitter.split(["विश्वासजनकम् आत्मानम्"])
    assert SEGMENT_SEPARATOR not in output
    assert output == "viSvAsa janakam AtmAnam"
    assert len(output.split()) == 3


def test_split_returns_slp1(monkeypatch: pytest.MonkeyPatch) -> None:
    splitter, fake = _splitter(monkeypatch)
    (output,) = splitter.split([DEVA])

    # The fake appends the IAST segment "iti"; SLP1 for it is "iti" too.
    assert output == to_slp1(_iast(DEVA), "iast") + " iti"
    assert output == "tat api gacCati iti"
    assert fake.seen == [SEGMENTATION_PREFIX + _iast(DEVA)]


def test_split_is_constructible_and_runnable_without_a_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    splitter, _ = _splitter(monkeypatch, model_id=None)
    assert splitter.model_id == SPLITTER_CANDIDATES[0]
    assert splitter.split([DEVA])


def test_split_restores_the_input_order_after_length_sorting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ["रामः", "रामः वनम् गच्छति एव", "रामः वनम्"]
    splitter, fake = _splitter(monkeypatch, batch_size=8)
    outputs = splitter.split(texts)

    assert outputs == [to_slp1(_iast(text), "iast") + " iti" for text in texts]
    # The model saw them longest-first, i.e. in an order the outputs do not preserve.
    assert fake.seen != [SEGMENTATION_PREFIX + _iast(text) for text in texts]
    assert sorted(fake.seen, key=len) == sorted(
        [SEGMENTATION_PREFIX + _iast(text) for text in texts], key=len
    )


def test_split_batches_at_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    splitter, fake = _splitter(monkeypatch, batch_size=2)
    splitter.split([DEVA, "रामः", "वनम्", "गच्छति", "एव"])
    assert [len(batch) for batch in fake.batches] == [2, 2, 1]


def test_split_uses_the_cache_on_the_second_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = SplitCache(tmp_path / "cache.jsonl")
    splitter, fake = _splitter(monkeypatch, cache=cache)

    first = splitter.split([DEVA])
    assert splitter.stats["n_model"] == 1
    assert splitter.stats["n_cache_hits"] == 0

    second = splitter.split([DEVA])
    assert second == first
    assert splitter.stats["n_model"] == 1
    assert splitter.stats["n_cache_hits"] == 1
    assert splitter.stats["n_calls"] == 2
    assert len(fake.batches) == 1


def test_split_persists_cache_entries_for_a_later_splitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "cache.jsonl"
    first, _ = _splitter(monkeypatch, cache=SplitCache(path))
    expected = first.split([DEVA])

    second, fake = _splitter(monkeypatch, cache=SplitCache(path))
    assert second.split([DEVA]) == expected
    assert fake.batches == []
    assert second.stats["n_model"] == 0


def test_split_deduplicates_repeated_texts_within_one_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    splitter, fake = _splitter(monkeypatch, cache=SplitCache(tmp_path / "cache.jsonl"))
    outputs = splitter.split([DEVA, DEVA])
    assert outputs[0] == outputs[1]
    assert len(fake.seen) == 1


def test_split_chunks_an_over_long_input(monkeypatch: pytest.MonkeyPatch) -> None:
    sentence = LONG_DEVA + " "
    text = (sentence * 40).strip()
    assert len(_iast(text).encode("utf-8")) > MAX_BYTES

    splitter, fake = _splitter(monkeypatch)
    (output,) = splitter.split([text])

    assert splitter.stats["n_chunked"] == 1
    assert len(fake.seen) > 1
    for prefixed in fake.seen:
        # What the model tokenises is the prefixed chunk plus an EOS token; it must fit
        # inside MAX_BYTES with room for that EOS, or `truncation=True` eats the tail.
        assert len(prefixed.encode("utf-8")) <= MAX_BYTES - 1
    # Every chunk's output is present, joined by spaces, in the original order.
    assert output.count("iti") == len(fake.seen)
    assert output.startswith(to_slp1(_iast(LONG_DEVA).split(" ")[0], "iast"))


def test_split_hard_splits_an_unbreakable_over_long_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "क" * (MAX_BYTES + 50)
    splitter, fake = _splitter(monkeypatch)
    splitter.split([text])

    assert splitter.stats["n_chunked"] == 1
    assert len(fake.seen) > 1
    for prefixed in fake.seen:
        assert len(prefixed.encode("utf-8")) <= MAX_BYTES - 1


def test_chunk_text_budget_leaves_room_for_the_prefix_and_eos() -> None:
    assert CHUNK_MAX_BYTES == MAX_BYTES - len(SEGMENTATION_PREFIX) - 1


def test_chunk_text_rejects_a_limit_smaller_than_one_character() -> None:
    with pytest.raises(ValueError, match="smaller than the first character"):
        chunk_text("क" * 10, limit=1)


def test_split_leaves_short_inputs_unchunked(monkeypatch: pytest.MonkeyPatch) -> None:
    splitter, _ = _splitter(monkeypatch)
    splitter.split([DEVA])
    assert splitter.stats["n_chunked"] == 0


def test_split_returns_empty_for_a_blank_input(monkeypatch: pytest.MonkeyPatch) -> None:
    splitter, fake = _splitter(monkeypatch)
    assert splitter.split(["   "]) == [""]
    assert fake.batches == []
    assert splitter.stats["n_model"] == 0


def test_stats_start_at_zero_and_record_model_time(monkeypatch: pytest.MonkeyPatch) -> None:
    splitter, _ = _splitter(monkeypatch)
    assert splitter.stats == {
        "n_calls": 0,
        "n_cache_hits": 0,
        "n_model": 0,
        "n_blank": 0,
        "n_duplicate": 0,
        "n_chunked": 0,
        "seconds_model": 0.0,
    }
    splitter.split([DEVA])
    assert splitter.stats["n_calls"] == 1
    assert splitter.stats["seconds_model"] >= 0.0


def test_source_id_falls_back_to_unknown_without_network(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import sanskrit_tok.sandhi.byt5 as byt5

    def boom(model_id: str) -> object:
        raise OSError("no network")

    monkeypatch.setattr(byt5, "_model_revision", boom)
    splitter = SandhiSplitter(model_id="some/model")
    with caplog.at_level(logging.WARNING):
        assert splitter.source_id == "some/model@unknown"
    assert any("revision" in record.message.lower() for record in caplog.records)


def test_source_id_is_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import sanskrit_tok.sandhi.byt5 as byt5

    calls: list[str] = []

    def revision(model_id: str) -> str:
        calls.append(model_id)
        return "deadbeef"

    monkeypatch.setattr(byt5, "_model_revision", revision)
    splitter = SandhiSplitter(model_id="some/model")
    assert splitter.source_id == "some/model@deadbeef"
    assert splitter.source_id == "some/model@deadbeef"
    assert calls == ["some/model"]


def test_stats_partition_the_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`n_calls == n_cache_hits + n_model + n_blank + n_duplicate`, exactly."""
    cache = SplitCache(tmp_path / "cache.jsonl")
    splitter, _ = _splitter(monkeypatch, cache=cache)
    splitter.split([DEVA])  # warms the cache for DEVA

    splitter.split([DEVA, "", DEVA, "रामः", "रामः", "   ", "वनम्"])

    stats = splitter.stats
    assert stats["n_calls"] == 8
    assert stats["n_blank"] == 2
    assert stats["n_duplicate"] == 2  # the second DEVA and the second रामः
    assert stats["n_cache_hits"] == 1  # DEVA, from the first call
    assert stats["n_model"] == 3  # DEVA (first call), रामः, वनम्
    assert stats["n_calls"] == (
        stats["n_cache_hits"] + stats["n_model"] + stats["n_blank"] + stats["n_duplicate"]
    )


def test_cache_is_flushed_after_every_batch_not_at_the_end_of_the_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run killed mid-call keeps every batch that completed before the failure.

    The split of a full corpus is a multi-hour background job; if the cache were written
    only when `split` returned, a crash in hour four would cost all four hours.
    """
    path = tmp_path / "cache.jsonl"
    cache = SplitCache(path)
    splitter = SandhiSplitter(cache=cache, batch_size=1)

    calls: list[list[str]] = []

    def explode_on_the_second_batch(batch: list[str]) -> list[str]:
        calls.append(list(batch))
        if len(calls) == 2:
            raise RuntimeError("simulated generation failure")
        return [text[len(SEGMENTATION_PREFIX) :] for text in batch]

    monkeypatch.setattr(splitter, "_generate_iast", explode_on_the_second_batch)

    with pytest.raises(RuntimeError, match="simulated generation failure"):
        splitter.split(["रामः", "रामः वनम् गच्छति एव"])

    assert len(calls) == 2
    # The first batch is on disk already — readable by a fresh cache, not merely buffered.
    reopened = SplitCache(path)
    assert len(reopened) == 1
    assert reopened.get("रामः वनम् गच्छति एव") == "rAmaH vanam gacCati eva"


# --- the real checkpoint (network-gated) ------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SANSKRIT_TOK_NETWORK_TESTS"),
    reason="set SANSKRIT_TOK_NETWORK_TESTS=1 to download and run the 2.3 GB checkpoint",
)
def test_real_model_splits_a_sandhied_sentence() -> None:
    splitter = SandhiSplitter()
    (output,) = splitter.split(["तदपि गच्छति"])
    assert len(output.split()) >= 2
    assert splitter.source_id.startswith(SPLITTER_CANDIDATES[0] + "@")
