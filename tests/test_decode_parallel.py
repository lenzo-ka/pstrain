"""Unit coverage for stable process-sharded decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pstrain.lib.pipeline import PipelineContext
from pstrain.lib.pipeline.tasks import build_pipeline
from pstrain.lib.setup import setup_project
from pstrain.lib.testing.decoder import Decoder, DecodingResult
from pstrain.lib.testing.test import _decode_files, _decode_shard, _DecodeConfig
from tests.conftest import requires_c_library

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"
FIXED_SEED_A = 243
FIXED_SEED_B = 17


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
def test_mini_arctic_dithered_decode_smoke_across_selected_shard_arrangements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke-test live dithered decode across selected shard arrangements.

    REFERENCE: one decoder traversing all ten mini_arctic utterances in natural
    order with a vocabulary-matched CI model trained by this checkout. AXES:
    reversed and rotated traversal; round-robin two- and three-shard assignments;
    swapped two-shard execution order; and a contiguous two-shard partition.
    Every nonempty shard gets exactly one decoder through our wrapper. Its
    captured process-wide native-init generation must remain unchanged across
    its utterances, detecting wrapper-mediated native reinitialization or
    whole-decoder replacement. It does not instrument hypothetical direct
    ``ps_reinit`` or ``ps_free``/``ps_init`` calls, which this path does not
    make. An
    A-to-B-to-A fixed-seed canary checks whether changed dither RNG segments
    affect this fixture's hypotheses. If none change, this is deliberately only
    an output smoke test, not evidence of general dithered-decode determinism.
    The effective-config query below checks the decoder's seed configuration
    value, not whether the frontend consumes that RNG stream. SILENT ON:
    feature-byte identity and frontend RNG consumption, larger corpora and
    models, other legal arrangements, other PocketSphinx versions, and
    nondithered decoding.
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
    # The default feature seed is -1 (automatic). Pin it before ci-1g training
    # so the trained fixture itself is reproducible across test runs.
    context = PipelineContext.from_config(
        project, cli_overrides={"features": {"seed": FIXED_SEED_A}}
    )
    assert build_pipeline(context).run("ci-1g", jobs=1) == 0

    model = context.model_dir("ci-1g")

    def set_model_seed(model_dir: Path, seed: int) -> None:
        feat_params = model_dir / "feat.params"
        feat_params.write_text(
            "\n".join(
                f"-seed {seed}" if line.startswith("-seed ") else line
                for line in feat_params.read_text().splitlines()
            )
            + "\n"
        )

    set_model_seed(model, FIXED_SEED_A)
    config = _DecodeConfig(
        model,
        context.shared_dir / "dictionary.dict",
        context.filler_dict,
        Path(__file__).parent.parent / "benchmarks" / "arctic" / "data" / "training-unigram.lm",
    )
    files = sorted((FIXTURE / "wav").glob("*.wav"))
    indexed = [(position, path.stem, path) for position, path in enumerate(files)]

    real_decoder = Decoder
    constructed: list[Decoder] = []
    decoded_paths: dict[int, list[Path]] = {}
    native_generations: dict[int, list[tuple[int, int]]] = {}

    class TrackedDecoder(real_decoder):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.effective_dither = self._lib.pstrain_decoder_config_int(self._decoder, b"dither")
            self.effective_seed = self._lib.pstrain_decoder_config_int(self._decoder, b"seed")
            self.native_init_generation = self._lib.pstrain_decoder_native_init_generation(
                self._decoder
            )
            constructed.append(self)
            decoded_paths[id(self)] = []
            native_generations[id(self)] = []

        def decode_file(self, path: Path) -> DecodingResult:
            decoded_paths[id(self)].append(path)
            before = self._lib.pstrain_decoder_native_init_generation(self._decoder)
            result = super().decode_file(path)
            after = self._lib.pstrain_decoder_native_init_generation(self._decoder)
            native_generations[id(self)].append((before, after))
            return result

    monkeypatch.setattr("pstrain.lib.testing.test.Decoder", TrackedDecoder)

    def decode(
        shards: list[list[tuple[int, str, Path]]],
        decode_seed: int,
        decode_config: _DecodeConfig = config,
    ) -> dict[str, str]:
        before = len(constructed)
        decoded = [item for shard in shards for item in _decode_shard(decode_config, shard)]
        shard_decoders = constructed[before:]
        nonempty_shards = [shard for shard in shards if shard]
        assert len(shard_decoders) == len(nonempty_shards)
        assert [decoded_paths[id(decoder)] for decoder in shard_decoders] == [
            [path for _, _, path in shard] for shard in nonempty_shards
        ]
        assert all(decoder.effective_dither == 1 for decoder in shard_decoders)
        assert all(decoder.effective_seed == decode_seed for decoder in shard_decoders)
        # Each wrapper captures the process-wide generation assigned by its
        # native construction. Reinitializing in place or replacing its native
        # decoder through the wrapper would expose a newer value here.
        assert all(
            native_generations[id(decoder)]
            == [(decoder.native_init_generation, decoder.native_init_generation)] * len(shard)
            for decoder, shard in zip(shard_decoders, nonempty_shards, strict=True)
        )
        assert all(result.success for _, _, result in decoded)
        return {utterance_id: result.hypothesis for _, utterance_id, result in decoded}

    natural = decode([indexed], FIXED_SEED_A)
    assert len(natural) == 10
    assert all(natural.values())

    arrangements = [
        [list(reversed(indexed))],
        [indexed[1:] + indexed[:1]],
        [indexed[worker::2] for worker in range(2)],
        list(reversed([indexed[worker::2] for worker in range(2)])),
        [indexed[worker::3] for worker in range(3)],
        [indexed[: len(indexed) // 2], indexed[len(indexed) // 2 :]],
    ]
    for shards in arrangements:
        assert decode(shards, FIXED_SEED_A) == natural

    # On one constant model root, decode fixed seed A, change only the in-place
    # seed to B, then restore A. The effective-config query checks that each
    # fresh decoder has the intended seed value; this black-box test does not
    # establish that the frontend consumes the corresponding RNG stream.
    set_model_seed(model, FIXED_SEED_B)
    perturbed = decode([indexed], FIXED_SEED_B)
    set_model_seed(model, FIXED_SEED_A)
    restored = decode([indexed], FIXED_SEED_A)
    assert restored == natural
    changed = {
        utterance_id
        for utterance_id, hypothesis in natural.items()
        if perturbed[utterance_id] != hypothesis
    }
    if not changed:
        pytest.xfail(
            f"mini_arctic changed 0/{len(natural)} hypotheses for fixed seed "
            f"{FIXED_SEED_A} -> {FIXED_SEED_B}, while restored seed {FIXED_SEED_A} "
            "reproduced all hypotheses; selected-arrangement equality is only a narrow "
            "output smoke test"
        )
    assert changed
