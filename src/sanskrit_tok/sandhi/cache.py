"""An append-only jsonl cache mapping a Sanskrit sentence to its sandhi-split form.

Splitting is the expensive step of Experiment 03: a 2.3 GB ByT5 model on a machine with
no CUDA GPU, run over ~137k sentences. The cache exists so that work is done once and
only once — a re-run, a crash, a second experiment that needs the same sentences, and the
tokenizer-training corpus builder all read the same file and skip whatever is already in
it (plan Global Constraints: "the cache makes re-runs skip finished work").

The format is one JSON object per line, `{"key", "input", "output"}`, appended and
flushed as results arrive rather than written once at the end, so an interrupted run
loses at most the in-flight batch. A run killed *mid-write* leaves a truncated final line;
that line is dropped with a WARNING on the next open rather than being an error, since the
alternative — refusing to load a 100k-entry cache over one partial record — would throw
away hours of work to save one sentence of it.

The key is the sha256 of the stripped input, so the same sentence with different
surrounding whitespace is one entry. `input` is stored beside it purely so the file is
readable and auditable by hand; lookups never use it.
"""

import hashlib
import json
import logging
from pathlib import Path
from types import TracebackType
from typing import TextIO

__all__ = ["SplitCache"]

logger = logging.getLogger(__name__)


class SplitCache:
    """A sentence -> split-sentence cache backed by one append-only jsonl file.

    Construction loads the whole file into memory (a few tens of MB at the corpus sizes
    this project uses) and leaves the file closed; the append handle is opened on the
    first `put`, so opening a cache read-only never creates a file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, str] = {}
        self._handle: TextIO | None = None
        self._load()

    # -- keys ---------------------------------------------------------------------

    @staticmethod
    def key(text: str) -> str:
        """sha256 of the stripped, UTF-8 encoded `text` — the cache key for a sentence."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    # -- loading ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        kept = 0
        with self.path.open(encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    logger.warning(
                        "%s: dropping a truncated last line (%d bytes); the run that "
                        "wrote it was interrupted mid-write",
                        self.path,
                        len(line),
                    )
                    continue
                raise
            self._entries[str(record["key"])] = str(record["output"])
            kept += 1
        logger.info("loaded %d split-cache entries from %s", kept, self.path)

    # -- reading and writing ------------------------------------------------------

    def get(self, text: str) -> str | None:
        """The cached split form of `text`, or `None` when it has not been split yet."""
        return self._entries.get(self.key(text))

    def put(self, text: str, output: str) -> None:
        """Record `output` as the split form of `text` and append it to the jsonl file."""
        key = self.key(text)
        self._entries[key] = output
        record = {"key": key, "input": text, "output": output}
        handle = self._open_for_append()
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        """Flush appended records to disk (callers flush once per completed batch)."""
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the append handle; the in-memory entries stay usable."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _open_for_append(self) -> TextIO:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    # -- dunders ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, text: str) -> bool:
        return self.key(text) in self._entries

    def __enter__(self) -> "SplitCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SplitCache(path={str(self.path)!r}, entries={len(self._entries)})"
