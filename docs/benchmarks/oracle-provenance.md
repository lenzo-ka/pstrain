# Arctic oracle provenance and reconstruction

The Arctic pin preserves the stock-SphinxTrain oracle's per-utterance scoring
rows in
[`oracle-sidecar.json`](../../evidence/arctic-pin/oracle-sidecar.json). Those
rows make the recorded comparison checkable: they retain the reference-word
and error counts for each decoded utterance, the aggregate scores, the
PocketSphinx and Python identities used for decoding, hashes of the decode
dictionary and language model, and hashes of the preserved model files.

That is tamper evidence for the recorded comparison, not independent
provenance for the stock training run. The sidecar does not name a
SphinxTrain source revision, build toolchain or build command. It does not
contain the stock training configuration, corpus archives or training-file
list, training log, decode command, or scoring command. Its original model
paths point outside the repository. Recomputing the rows from preserved model
bytes would check decoding, but would not reconstruct the models.

## Procedure for an independently derived stock arm

The following is the work required to create a comparable arm. It is not a
record of a procedure executed for the checked-in oracle.

1. Pin a clean stock SphinxTrain checkout by full commit and record the
   compiler, compiler version, build flags, platform, and hashes of the
   resulting programs. Development records name
   `694c10099a50d11fd31ca824044bbe957ba650bd` as a stock revision that was
   rebuilt successfully on a narrower CI-training rung. The oracle sidecar
   does not bind its models to that revision, so it is a reconstruction
   candidate, not established oracle provenance. If another revision is
   used, record why and keep its full identity with the result.
2. Reconstruct the CMU Arctic SLT training project from the corpus identities
   and 1,043-utterance training-file-list hash in the
   [pin record](../../evidence/arctic-pin/record.json). The checked-in record
   supplies archive, transcript, and file-list hashes, but not a stock project
   tree or a mapping from those inputs to a complete `sphinx_train.cfg`.
   Recover or recreate that configuration, then check it in or preserve it
   verbatim with a digest. Do not treat a configuration inferred only from
   final model dimensions as the producing configuration.
3. Produce separate stock multipron-off and multipron-on models, recording
   every effective stock setting, stage input, realized pass count, omitted
   utterance, and output-model hash. The historical development notes refer
   to external `run.sh` and `etc/sphinx_train.cfg` files and identify
   `CFG_MULTIPRON_TRAINING=yes` as the on-arm change, but those files are not
   in this repository. Until they are recovered or replaced by a complete,
   reviewed configuration, the historical oracle recipe remains incomplete.
4. For a pstrain comparison intended to mirror stock alignment behavior, set
   `training.retry_beam_factor=1`, `training.failed_alignment=omit`, and
   `training.optional_final_silence=false`, as required by the
   [parity and deviations register](../design/parity-and-deviations.md).
   Also select the stock-compatible inventory and sharding postures described
   there, and pin the front-end difference rather than assuming the two arms
   consume identical feature bytes. These are comparison controls for the
   pstrain arm, not claims about stock configuration keys.
5. Decode the reconstructed models under the pin's recorded PocketSphinx,
   language-model, dictionary, filler-dictionary, beam, language-weight, and
   word-insertion-penalty identities. Preserve the exact decode and scoring
   commands. Emit per-utterance rows in the sidecar schema and compare both
   rows and model hashes with the checked-in oracle. A mismatch is a new
   measured arm unless the missing producing-run provenance is recovered; it
   must not silently replace or relabel the preserved oracle.

This procedure becomes a reproduction of the checked-in oracle only when the
missing source/configuration/toolchain link is supplied and the resulting
models and rows agree. At present it is a route to an independently derived,
comparable stock baseline. The preserved oracle remains an attested historical
measurement rather than a fully reproducible build.
