# Provenance and licensing of `csrc/`

The C substrate in this directory combines vendored CMU Sphinx code with pstrain
modifications and new code. Two licenses apply; see below.

## Vendored CMU Sphinx code — modified (CMU BSD + pstrain BSD 2-Clause)

Most of `csrc/` is derived from the **CMU Sphinx** project — specifically
**SphinxTrain**, **SphinxBase**, and **Sphinx-3**:

- `libs/libio/`, `libs/libcommon/`, `libs/libclust/`, `libs/libmllr/`,
  `libs/libmodinv/` — SphinxTrain libraries.
- `libs/libsphinxbase/` — SphinxBase (allocator, cmd_ln, logmath, MFCC
  front-end, feature transforms, and a bundled LAPACK/BLAS-lite fallback).
- `programs/` — the SphinxTrain command-line programs, plus the Sphinx-3
  `sphinx3_align` forced aligner.

These files **have been modified** as part of pstrain (e.g. symbol namespacing for
the Sphinx-3 aligner, `#ifdef PSTRAIN_LIBRARY_BUILD` guards that strip `main()` so
program sources can be linked into the shared library, portability and
build-integration fixes). As a result they are effectively **dual-licensed**:

- The original portions remain under the **CMU Sphinx BSD-style license**
  (Copyright (c) 1999–2016 Carnegie Mellon University). Its full text is in
  [`LICENSE.sphinx`](LICENSE.sphinx), and the per-file CMU copyright headers are
  retained as required by that license.
- The pstrain **modifications** are Copyright (c) 2026 Kevin Lenzo and are licensed
  under the **BSD 2-Clause license** (repository root [`LICENSE`](../LICENSE)).

Redistribution must satisfy both licenses; both are permissive BSD-style
and compatible.

Upstream: https://github.com/cmusphinx/sphinxtrain

## New pstrain code — BSD 2-Clause

`libs/libpstrain/` (the `pstrain_*.c` / `pstrain_*.h` session-wrapper layer that exposes a
simplified, CFFI-friendly API over the SphinxTrain internals) and the pstrain build
system are original work, Copyright (c) 2026 Kevin Lenzo, distributed under the
BSD 2-Clause license in the repository root [`LICENSE`](../LICENSE).
