# Failed-alignment policy

## Deliberate vendored divergence

Decision date: 2026-08-14.

Vendored SphinxTrain reports each failed Baum–Welch utterance as ignored,
discards its update, and continues training. Pstrain deliberately defaults to
`training.failed_alignment: recover`: it retries forward-final-state pruning
failures once with the configured wider beam and aborts if recovery fails. The
default remains loud because silently removing training examples can make a
successful run train on unintended data.

The schema-owned positions are:

- `recover` (default): retry once, then abort if the alignment still fails;
- `abort`: do not retry and abort immediately; and
- `omit`: report the named utterance, exclude its failed update from the
  accumulators, and continue, reproducing upstream's observable behavior.

This setting changes only the disposition of failed training alignments. It
does not change beams, metrics, evaluation, or any unrelated default.
