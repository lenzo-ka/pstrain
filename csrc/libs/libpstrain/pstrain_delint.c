/* ====================================================================
 * Copyright (c) 1999-2016 Carnegie Mellon University.  All rights
 * reserved.
 *
 * This file is derived from CMU SphinxTrain/SphinxBase sources.
 * Modifications for pstrain are Copyright (c) 2026 Kevin Lenzo and are
 * licensed under the BSD 2-Clause license (see LICENSE at the repository
 * root). The repository history records the modifications.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * This work was supported in part by funding from the Defense Advanced
 * Research Projects Agency and the National Science Foundation of the
 * United States of America, and the CMU Sphinx Speech Consortium.
 *
 * THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
 * NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 */
#include "pstrain_delint.h"

#include <s3/common.h>
#include <s3/model_def_io.h>
#include <s3/s3mixw_io.h>

#include <sphinxbase/matrix.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>

#include <sys_compat/file.h>

#include <stdio.h>
#include <string.h>

/* Forward declaration - from delint/main.c */
extern int32 smooth_mixw(float32 ****out_mixw,
                         float32 ***mixw_acc_a,
                         float32 ***mixw_acc_b,
                         uint32 n_mixw,
                         uint32 n_feat,
                         uint32 n_gau,
                         model_def_t *mdef,
                         float32 cilambda,
                         int32 maxiter);

static int
rd_param(uint32 *idx,
         const char **accumdirs,
         float32 ****out_mixw_acc,
         uint32 *out_n_mixw,
         uint32 *out_n_feat,
         uint32 *out_n_gau)
{
    char fn[MAXPATHLEN+1];
    const char *accum_dir;
    uint32 i;

    i = *idx;
    accum_dir = accumdirs[i];

    snprintf(fn, MAXPATHLEN, "%s/mixw_counts", accum_dir);

    E_INFO("Reading %s\n", fn);

    if (s3mixw_read(fn,
                    out_mixw_acc,
                    out_n_mixw,
                    out_n_feat,
                    out_n_gau) != S3_SUCCESS) {
        return S3_ERROR;
    }

    ++(*idx);

    return S3_SUCCESS;
}

int pstrain_delint(const char *moddeffn,
               const char *mixwfn,
               const char **accumdirs,
               float32 cilambda,
               int32 maxiter)
{
    model_def_t *mdef = NULL;
    float32 ***mixw_acc_in = NULL;
    float32 ***mixw_acc_a = NULL;
    float32 ***mixw_acc_b = NULL;
    float32 ***mixw = NULL;
    uint32 n_mixw, n_feat, n_gau;
    uint32 i;
    int ret = -1;

    if (moddeffn == NULL) {
        E_ERROR("Must specify model definition file\n");
        return -1;
    }

    if (mixwfn == NULL) {
        E_ERROR("Must specify output mixture weight file\n");
        return -1;
    }

    if (accumdirs == NULL || accumdirs[0] == NULL || accumdirs[1] == NULL) {
        E_ERROR("Must specify at least 2 accumulator directories\n");
        return -1;
    }

    /* Read model definition */
    if (model_def_read(&mdef, moddeffn) != S3_SUCCESS) {
        E_ERROR("Failed to read model definition from %s\n", moddeffn);
        return -1;
    }

    /* Read first two accumulator directories */
    i = 0;
    if (rd_param(&i, accumdirs, &mixw_acc_a, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
        E_ERROR("Failed to read first accumulator directory\n");
        goto cleanup;
    }

    if (rd_param(&i, accumdirs, &mixw_acc_b, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
        E_ERROR("Failed to read second accumulator directory\n");
        goto cleanup;
    }

    /* Read additional directories (must be even number) */
    while (accumdirs[i] != NULL) {
        if (rd_param(&i, accumdirs, &mixw_acc_in, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
            E_ERROR("Failed to read accumulator directory %d\n", i);
            goto cleanup;
        }

        /* Accumulate into A buffer */
        accum_3d(mixw_acc_a, mixw_acc_in, n_mixw, n_feat, n_gau);
        ckd_free_3d((void ***)mixw_acc_in);
        mixw_acc_in = NULL;

        /* Must have even number */
        if (accumdirs[i] == NULL) {
            E_ERROR("Must specify even number of accumulator directories\n");
            goto cleanup;
        }

        if (rd_param(&i, accumdirs, &mixw_acc_in, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
            E_ERROR("Failed to read accumulator directory %d\n", i);
            goto cleanup;
        }

        /* Accumulate into B buffer */
        accum_3d(mixw_acc_b, mixw_acc_in, n_mixw, n_feat, n_gau);
        ckd_free_3d((void ***)mixw_acc_in);
        mixw_acc_in = NULL;
    }

    /* Run deleted interpolation */
    if (smooth_mixw(&mixw,
                    mixw_acc_a, mixw_acc_b,
                    n_mixw, n_feat, n_gau,
                    mdef, cilambda, maxiter) != S3_SUCCESS) {
        E_ERROR("Deleted interpolation failed\n");
        goto cleanup;
    }

    /* Write output */
    E_INFO("Writing %s\n", mixwfn);
    if (s3mixw_write(mixwfn, mixw, n_mixw, n_feat, n_gau) != S3_SUCCESS) {
        E_ERROR("Failed to write mixture weights to %s\n", mixwfn);
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (mdef != NULL)
        model_def_free(mdef);
    if (mixw_acc_in != NULL)
        ckd_free_3d((void ***)mixw_acc_in);
    /* Note: mixw_acc_a and mixw_acc_b are freed by smooth_mixw */

    return ret;
}
