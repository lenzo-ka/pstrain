# Baum-Welch normalization policy

Baum-Welch normalization keeps two variance representations with a strict
ownership invariant:

- `raw_var` is the direct `V/N` (or one-pass `E[x²] - E[x]²`) result. It is
  serialized without flooring or reciprocal conversion.
- `gauden.var` is evaluation-only. It receives a copy of `raw_var`, applies
  the `1e-4` load/evaluation floor, and is then converted to reciprocals by
  Gaussian precomputation.

An unobserved Gaussian has no normalization result. Every BW caller must choose
one of two explicit policies:

- `zero` clears its output mean and raw variance, matching upstream `norm`'s
  fresh-output allocation. All pipeline stages select this policy for parity.
- `retain` preserves the input Gaussian. This can be safer for sparse or
  exploratory training, but it is a deliberate divergence and never a default.

Documenting `retain` as an opt-in alternative, and potentially adding the same
choice upstream, is an `UPSTREAM.md` contribution candidate. Artifact parity
requires `zero` unless upstream itself adopts a different explicit policy.
