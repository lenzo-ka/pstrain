"""Unit coverage for stable process-sharded decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pstrain.lib.testing.decoder import DecodingResult
from pstrain.lib.testing.test import _decode_files, _decode_shard, _DecodeConfig


def _config() -> _DecodeConfig:
    return _DecodeConfig(Path("model"), Path("dict"), None, None)


def test_decode_shard_constructs_one_decoder_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class FakeDecoder:
        def __init__(self, **kwargs: Any) -> None:
            self.paths: list[Path] = []
            self.closed = False
            instances.append(self)

        def decode_file(self, path: Path) -> DecodingResult:
            self.paths.append(path)
            return DecodingResult(path.stem, path.stem.upper(), True)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("pstrain.lib.testing.test.Decoder", FakeDecoder)
    shard = [(0, "a", Path("a.wav")), (2, "c", Path("c.wav"))]

    assert [item[1] for item in _decode_shard(_config(), shard)] == ["a", "c"]
    assert len(instances) == 1
    assert instances[0].paths == [Path("a.wav"), Path("c.wav")]
    assert instances[0].closed


def test_parallel_merge_is_canonical_and_uses_distinct_executor_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate process-shard merging against this test's mocked serial decode.

    REFERENCE: one-worker pstrain traversal using the same fake decoder, fake
    hypotheses, model/dictionary placeholders, and no live aligner or scorer.
    AXIS: worker count (one versus two) and reversed per-shard completion order.
    SILENT ON: real decoder or model correctness, defects shared with serial
    traversal, other orderings, architecture, and arithmetic.
    """
    executor_calls: list[int] = []

    class ImmediateFuture:
        def __init__(self, value: Any) -> None:
            self.value = value

        def result(self) -> Any:
            return self.value

    class ImmediateExecutor:
        def __init__(self, max_workers: int, mp_context: Any) -> None:
            executor_calls.append(max_workers)

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def submit(self, fn: Any, *args: Any) -> ImmediateFuture:
            return ImmediateFuture(fn(*args))

    def fake_shard(
        config: _DecodeConfig, shard: list[tuple[int, str, Path]]
    ) -> list[tuple[int, str, DecodingResult]]:
        return [
            (position, utt_id, DecodingResult(utt_id, f"hyp-{utt_id}", True))
            for position, utt_id, _ in reversed(shard)
        ]

    monkeypatch.setattr("pstrain.lib.testing.test.ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr("pstrain.lib.testing.test._decode_shard", fake_shard)
    files = [(name, Path(f"{name}.wav")) for name in ("a", "b", "c", "d", "e")]

    serial = _decode_files(_config(), files, 1)
    parallel = _decode_files(_config(), files, 2)

    assert executor_calls == [2]
    assert [utt_id for utt_id, _ in parallel] == ["a", "b", "c", "d", "e"]
    assert {utt_id: result.hypothesis for utt_id, result in parallel} == {
        utt_id: result.hypothesis for utt_id, result in serial
    }
