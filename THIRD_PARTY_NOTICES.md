# Third-party notices

This file records the notices for third-party source carried in `csrc/`. These
CMU components are modified vendored code, so both licenses apply in layers:
the base remains Copyright (c) Carnegie Mellon University under
`csrc/LICENSE.sphinx`, while the pstrain modifications are Copyright (c) 2026
Kevin Lenzo under the BSD 2-Clause license in [LICENSE](LICENSE). The git
repository history is the authoritative record of what changed.

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

The vendored file does not state a separate copyright sentence. Its applicable
LAPACK routine attribution, version, and date notices are reproduced here
verbatim (routine names are supplied as headings for context):

### IEEECK

> -- LAPACK auxiliary routine (version 3.0) --
> Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
> Courant Institute, Argonne National Lab, and Rice University
> June 30, 1998

### ILAENV

> -- LAPACK auxiliary routine (version 3.0) --
> Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
> Courant Institute, Argonne National Lab, and Rice University
> June 30, 1999

### SPOSV

> -- LAPACK driver routine (version 3.0) --
> Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
> Courant Institute, Argonne National Lab, and Rice University
> March 31, 1993

### SPOTF2

> -- LAPACK routine (version 3.0) --
> Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
> Courant Institute, Argonne National Lab, and Rice University
> February 29, 1992

### SPOTRF

> -- LAPACK routine (version 3.0) --
> Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
> Courant Institute, Argonne National Lab, and Rice University
> March 31, 1993

### SPOTRS

> -- LAPACK routine (version 3.0) --
> Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
> Courant Institute, Argonne National Lab, and Rice University
> March 31, 1993

This component is a conditional, f2c-translated LAPACK fallback.

## `f2c_lite`

The vendored
[`csrc/libs/libsphinxbase/util/f2c_lite.c`](csrc/libs/libsphinxbase/util/f2c_lite.c)
starts with includes and contains no copyright or license header. It implements
a subset of the f2c compatibility runtime; the following permissive notice is
reproduced verbatim from the `Notice` file in the recognized upstream Netlib
`libf2c.zip` distribution:

> Copyright 1990 - 1997 by AT&T, Lucent Technologies and Bellcore.
>
> Permission to use, copy, modify, and distribute this software
> and its documentation for any purpose and without fee is hereby
> granted, provided that the above copyright notice appear in all
> copies and that both that the copyright notice and this
> permission notice and warranty disclaimer appear in supporting
> documentation, and that the names of AT&T, Bell Laboratories,
> Lucent or Bellcore or any of their entities not be used in
> advertising or publicity pertaining to distribution of the
> software without specific, written prior permission.
>
> AT&T, Lucent and Bellcore disclaim all warranties with regard to
> this software, including all implied warranties of
> merchantability and fitness.  In no event shall AT&T, Lucent or
> Bellcore be liable for any special, indirect or consequential
> damages or any damages whatsoever resulting from loss of use,
> data or profits, whether in an action of contract, negligence or
> other tortious action, arising out of or in connection with the
> use or performance of this software.

This component is the conditional lightweight f2c runtime used by the LAPACK
fallback.
