# DECODE-STATE report — 2026-08-12

## Status

Investigation in progress. No decoder configuration, reset behavior, or production
code has been changed.

Branch: `probe/decode-state`

## Reproduction boundary and import provenance

The established `clb/arctic_b0438` result and the 1-of-20 contiguous-partition
difference are accepted from
`/Volumes/experiments/pstrain-parity/decode-par-report-2026-08-12.md`; live CMN is
already ruled out and was not re-derived.

A dedicated virtual environment was created at
`/Volumes/experiments/pstrain-parity/decode-state-tree/.venv` with PocketSphinx
5.1.1. `scripts/decode_state_probe.py --provenance` resolved both parent and
spawned-worker imports as follows:

| Process | `pstrain` | benchmark module | PocketSphinx |
|---|---|---|---|
| parent | `/Volumes/experiments/pstrain-parity/decode-state-tree/pstrain/__init__.py` | `/Volumes/experiments/pstrain-parity/decode-state-tree/pstrain/benchmarks/arctic.py` | worktree `.venv`, version 5.1.1 |
| spawned worker | `/Volumes/experiments/pstrain-parity/decode-state-tree/pstrain/__init__.py` | `/Volumes/experiments/pstrain-parity/decode-state-tree/pstrain/benchmarks/arctic.py` | worktree `.venv`, version 5.1.1 |

No spawned or parent import resolved to the original checkout.

## Named state and mechanism

Pending empirical isolation.

## Intended behavior versus driver artifact

Pending.

## Owner questions

Pending.

## Reordering sensitivity and bootstrap implications

Pending.

## Deliberately untouched

- Production decoder configuration and reset behavior.
- Transcript ordering, model files, language model, dictionary, and scoring.
