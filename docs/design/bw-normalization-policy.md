# Baum-Welch normalization policy

Baum-Welch normalization keeps two variance representations with a strict
ownership invariant:

- `raw_var` is the direct `V/N` (or one-pass `E[x²] - E[x]²`) result. It is
  serialized without flooring or reciprocal conversion.
- `gauden.var` is evaluation-only. It receives a copy of `raw_var`, applies
  the `1e-4` load/evaluation floor, and is then converted to reciprocals by
  Gaussian precomputation.

An unobserved parameter cell has no normalization result. Every BW caller must
choose one of two explicit policies:

- `zero` clears output means, raw variances, mixture weights, and transition
  rows, matching upstream `norm`'s fresh accumulator/output allocation. All
  pipeline stages select this policy for parity.
- `retain` preserves the corresponding input values. This can be safer for sparse or
  exploratory training, but it is a deliberate divergence and never a default.

Fallback-tracked CI senones' Gaussians and mixture weights are always retained
when they receive no posterior mass, including under `zero`. Graph membership
marks these cells, so a branch that is not selected in one pass remains usable
in a later pass. Transition matrices have no senone-level fallback mapping.
Linear/parity runs do not mark fallback senones and therefore retain upstream
zero semantics.

On reload, an all-zero mixture row is normalized (and reports failure), then
every cell receives the configured mixture floor and the row is normalized
again. It therefore becomes uniform for evaluation, matching upstream senone
loading while preserving exact-zero training artifacts.

For a given loaded float, the evaluation floor (`1e-4`) and reciprocal formula
are unchanged. Evaluation is not invariant across a save/reload boundary:
lossless serialization means the next pass sees the original sub-floor value,
where the former path saw a floored, round-tripped value. That difference is the
intended correction to the lossy path.

Documenting `retain` as an opt-in alternative, and potentially adding the same
choice upstream, is an `UPSTREAM.md` contribution candidate. Artifact parity
requires `zero` unless upstream itself adopts a different explicit policy.

Follow-up: add a discriminating decode test proving PocketSphinx handles
zero/unfloored exported artifacts the same way it handles upstream artifacts.
Upstream norm writes the same zeros, so parity is currently by convention rather
than direct decode coverage.
