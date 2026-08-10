# Third-party notices

This file records the notices for third-party source carried in `csrc/`. The
source files remain subject to their own notices in addition to the license for
new pstrain code in [LICENSE](LICENSE).

## CMU SphinxTrain, SphinxBase, and Sphinx-3

Copyright (c) 1999-2016 Carnegie Mellon University. All rights reserved.

License: CMU BSD-style two-clause license. The full license text is in
[csrc/LICENSE.sphinx](csrc/LICENSE.sphinx).

The vendored and derived CMU code includes SphinxTrain libraries and programs,
the SphinxBase code under `csrc/libs/libsphinxbase/`, the Sphinx-3 forced
aligner, and CMU-derived implementation code in `csrc/libs/libpstrain/`.

The CMU notice, reproduced in full, is:

> Copyright (c) 1999-2016 Carnegie Mellon University.  All rights
> reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions
> are met:
>
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in
>    the documentation and/or other materials provided with the
>    distribution.
>
> This work was supported in part by funding from the Defense Advanced
> Research Projects Agency and the National Science Foundation of the
> United States of America, and the CMU Sphinx Speech Consortium.
>
> THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
> ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
> THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
> PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
> NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
> LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
> DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
> THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## LAPACK `slapack_lite`

Copyright/attribution line in the vendored routine headers: “Univ. of
Tennessee, Univ. of California Berkeley, NAG Ltd., Courant Institute, Argonne
National Lab, and Rice University.” The file does not state a separate
copyright sentence.

License/notice: historical LAPACK 3.0 routine notices embedded in the generated
C source. The full notices, including the routine version, dates, attribution,
and documentation, remain in
[`csrc/libs/libsphinxbase/util/slapack_lite.c`](csrc/libs/libsphinxbase/util/slapack_lite.c).

This component is a conditional, f2c-translated LAPACK fallback.

## `f2c_lite`

Copyright line: none is stated in the vendored file header.

License/notice: generated f2c compatibility-runtime source notice. The full
header notice remains in
[`csrc/libs/libsphinxbase/util/f2c_lite.c`](csrc/libs/libsphinxbase/util/f2c_lite.c),
including its statement that it is generated code and the embedded f2c
subscript-checking notice.

This component is the conditional lightweight f2c runtime used by the LAPACK
fallback.
