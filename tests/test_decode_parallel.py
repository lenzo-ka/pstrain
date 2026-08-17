"""Unit coverage for stable process-sharded decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pstrain.lib.pipeline import PipelineContext
from pstrain.lib.pipeline.tasks import build_pipeline
from pstrain.lib.setup import setup_project
from pstrain.lib.testing.decoder import DecodingResult
from pstrain.lib.testing.test import _decode_files, _decode_shard, _DecodeConfig
from tests.conftest import requires_c_library

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


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


@requires_c_library
def test_dithered_decode_hypotheses_are_independent_of_shard_order(tmp_path: Path) -> None:
    """Gate live dithered decode against the natural one-decoder traversal.

    REFERENCE: one decoder traversing all ten mini_arctic utterances in natural
    order with a vocabulary-matched CI model trained by this checkout. AXIS:
    reversed traversal and legal round-robin two- and three-shard assignments,
    each shard using a fresh decoder. SILENT ON: feature-byte identity, larger
    corpora and models, other PocketSphinx versions, and nondithered decoding.
    """
    project = tmp_path / "project"
    setup_project(
        project,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )
    context = PipelineContext.from_config(project)
    assert build_pipeline(context).run("ci-1g", jobs=1) == 0

    model = context.model_dir("ci-1g")
    assert "-dither yes" in (model / "feat.params").read_text().splitlines()
    config = _DecodeConfig(
        model,
        context.shared_dir / "dictionary.dict",
        context.filler_dict,
        Path(__file__).parent.parent / "benchmarks" / "arctic" / "data" / "training-unigram.lm",
    )
    files = sorted((FIXTURE / "wav").glob("*.wav"))
    indexed = [(position, path.stem, path) for position, path in enumerate(files)]

    def decode(shards: list[list[tuple[int, str, Path]]]) -> dict[str, str]:
        decoded = [item for shard in shards for item in _decode_shard(config, shard)]
        assert all(result.success for _, _, result in decoded)
        return {utterance_id: result.hypothesis for _, utterance_id, result in decoded}

    natural = decode([indexed])
    assert len(natural) == 10
    assert all(natural.values())

    arrangements = [
        [list(reversed(indexed))],
        [indexed[worker::2] for worker in range(2)],
        [indexed[worker::3] for worker in range(3)],
    ]
    for shards in arrangements:
        assert decode(shards) == natural
