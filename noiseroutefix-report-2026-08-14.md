# PR #82 complete-model front-end contract report

## Outcome

- **[measured]** Commit `b5ddd27` records the red contract state: the targeted run
  had exactly two failures, for decoding and alignment without `feat.params`, and
  38 passes.
- **[measured]** Commit `e4c9ddc` makes those tests green by enforcing the
  complete-model contract at decoder and aligner construction and by sharing the
  same check with packaging.
- **[read]** The missing-file error names `feat.params`, its full path and model
  directory, and explains that an undefined decode-time front end can silently
  differ from training in feature shape and basis.
- **[read]** The training/BW five-file contract remains unchanged, while
  `MODEL_FILES_COMPLETE_REQUIRED` declares that complete consumers additionally
  require `feat.params`.

## Consumer inventory and boundary

- **[read]** `pstrain.lib.testing.decoder.Decoder` loads a PocketSphinx acoustic
  model for unconstrained decoding; it expects a complete model.
- **[read]** `pstrain.lib.testing.test.test_model`, `pstrain.cli.test`, the Arctic
  benchmark, and `scripts/compare_multipron_alignments.py` reach model loading
  through `Decoder`; they therefore expect a complete model rather than defining
  separate model contracts.
- **[read]** `pstrain.lib.alignment.native.Aligner` loads acoustic parameters and
  initializes the forced-alignment front end; it expects a complete model.
- **[read]** `pstrain.lib.alignment.core`, `pstrain.lib.alignment.batch`, and
  `pstrain.cli.align` reach model loading through `Aligner`; they therefore expect
  a complete model rather than defining separate model contracts.
- **[read]** `pstrain.lib.steps.package.package_model` consumes a trained model for
  distribution and already required `feat.params`; it expects a complete model.
- **[read]** The packaging pipeline task reaches model loading through
  `package_model`; it expects a complete model.
- **[read]** `BWModel.load`, `run_bw_training`, Gaussian splitting, flat-model
  initialization, CI/CD initialization and parameter copying load or transform
  model directories during training; they operate on models under construction.
- **[read]** model comparison and model file-type discovery inspect model
  directories but do not initialize a decode/align front end; they retain the
  construction-time five-file contract.
- **[inference]** Path-only `Model`/`PipelineContext.model_dir` helpers are not
  consumers because they identify directories without loading model content.
- **[measured]** A positive test constructs the five training/BW files without
  `feat.params` and still classifies the directory as a model, proving the scoped
  check does not make project initialization or training construction globally
  fatal.

## Routing, precedence, and gate discrimination

- **[read]** `remove_noise` remains routed from `FeatureConfig` into
  `pstrain_decoder_config_t` and then into the PocketSphinx config before
  `ps_init()`.
- **[measured]** With `-remove_noise` omitted from an otherwise present
  `feat.params`, deleting the native assignment makes the `True` arm read the
  PocketSphinx default `0` and fail.
- **[read]** The test docstring states the asymmetry: the `False` arm is unchanged
  when that assignment is deleted, so it does not gate the missing-assignment
  defect; it gates always-true and inverted assignments.
- **[read]** Every other direct-routing assertion touched here uses a value that
  differs from the value observed if its assignment is deleted. In particular,
  the `cmn` probe is `batch`, not PocketSphinx's `live` default.
- **[measured]** Adding `feat.params` to the previously incomplete decoder fixture
  initially made five direct-routing assertions fail because the file won over
  the profile. The tests were corrected to omit only the key being probed.
- **[read]** Configuration documentation now declares that the trained model's
  `feat.params` is authoritative over the schema profile because it records how
  the training features were made and PocketSphinx reads it after pstrain's
  pre-initialization assignments.
- **[read]** The 6/19 statement is limited to decoder live configuration and is
  conditional: those six direct assignments govern only when `feat.params` does
  not override them.
- **[read]** The remaining 13 decoder fields are supplied by the now-required
  `feat.params`, so PocketSphinx defaults for them are unreachable in supported
  complete-model decode/align/package paths.
- **[inference]** Those 13 defaults become reachable again if a complete-model
  consumer is allowed to omit `feat.params` or bypass the shared contract.
- **[read]** The C contract stub now states that it models direct no-file
  configuration only and is silent on `feat.params` precedence and real
  PocketSphinx front-end state.

## Artifact finding

- **[measured]** `tests/fixtures/multipron_final_state/model` lacked
  `feat.params`; making absence fatal exposed it immediately.
- **[read]** The fixture now carries a complete training front-end record rather
  than relying on PocketSphinx defaults.
- **[inference]** This was an incomplete complete-model fixture, not a legitimate
  construction-time consumer, because it is used to initialize live decoders.

## Gates

- **[measured]** Targeted red at `b5ddd27`: 2 failed, 38 passed.
- **[measured]** Targeted green at `e4c9ddc`: 40 passed.
- **[measured]** Native C tests: 6/6 passed with nine-way parallelism.
- **[measured]** Native floating-point contraction scan: 39 artifact
  architectures passed.
- **[measured]** Full strict Python suite: 539 passed with nine workers.
- **[measured]** `ruff check pstrain tests`: passed.
- **[measured]** `mypy pstrain`: passed.
- **[measured]** The first sandboxed full-suite attempt produced 523 passes and
  16 process-control failures caused by denied process/semaphore/`ps` operations;
  the identical permitted run passed 539/539.

## Declined work

- **[read]** The remaining 13 `FeatureConfig` fields were not routed directly;
  doing so would be a separate configuration-contract change forbidden by this
  lane, and complete consumers now obtain them from the authoritative file.
- **[read]** No metric, evaluator, unrelated default, training-stage model
  contract, or benchmark input was changed.
- **[read]** The branch was not merged.
