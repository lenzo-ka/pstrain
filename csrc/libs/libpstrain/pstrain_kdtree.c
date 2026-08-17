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
#include "pstrain_kdtree.h"

#include <s3/common.h>
#include <s3/s3gau_io.h>
#include <s3/gauden.h>
#include <s3/kdtree.h>

#include <stdio.h>

int pstrain_kdtree_build(const char *meanfn,
                     const char *varfn,
                     const char *outfn,
                     float32 threshold,
                     int32 depth,
                     int32 absolute)
{
    vector_t ***means = NULL, ***variances = NULL;
    uint32 n_mgau, n_feat, n_density;
    uint32 r_n_mgau, r_n_feat, r_n_density;
    uint32 *veclen = NULL, *r_veclen = NULL;
    uint32 i;
    kd_tree_node_t **root = NULL;
    int ret = -1;

    if (meanfn == NULL || varfn == NULL) {
        E_ERROR("Must specify both meanfn and varfn\n");
        return -1;
    }

    /* Read means */
    if (s3gau_read(meanfn, &means, &n_mgau,
                   &n_feat, &n_density, &veclen) != S3_SUCCESS) {
        E_ERROR("Failed to read means from %s\n", meanfn);
        goto cleanup;
    }

    /* Read variances */
    if (s3gau_read(varfn, &variances, &r_n_mgau,
                   &r_n_feat, &r_n_density, &r_veclen) != S3_SUCCESS) {
        E_ERROR("Failed to read variances from %s\n", varfn);
        goto cleanup;
    }

    /* Validate dimensions */
    if (n_mgau != r_n_mgau) {
        E_ERROR("Number of GMMs in variances doesn't match means: %d != %d\n",
                r_n_mgau, n_mgau);
        goto cleanup;
    }
    if (n_mgau != 1) {
        E_ERROR("Only semi-continuous models are currently supported\n");
        goto cleanup;
    }
    if (n_density != r_n_density) {
        E_ERROR("Number of Gaussians in variances doesn't match means: %d != %d\n",
                r_n_density, n_density);
        goto cleanup;
    }
    if (n_feat != r_n_feat) {
        E_ERROR("Number of feature streams in variances doesn't match means: %d != %d\n",
                r_n_feat, n_feat);
        goto cleanup;
    }
    for (i = 0; i < n_feat; ++i) {
        if (veclen[i] != r_veclen[i]) {
            E_ERROR("Size of feature stream %d in variances doesn't match means: %d != %d\n",
                    i, r_veclen[i], veclen[i]);
            goto cleanup;
        }
    }

    gauden_floor_variance_array(variances, r_n_mgau, r_n_feat,
                                r_n_density, r_veclen,
                                GAUDEN_EVAL_VAR_FLOOR);

    /* Build one kd-tree for each feature stream */
    root = ckd_calloc(n_feat, sizeof(*root));
    for (i = 0; i < n_feat; ++i) {
        root[i] = build_kd_tree(means[0][i], variances[0][i],
                                n_density, veclen[i],
                                threshold, depth, absolute);
    }

    /* Write output if requested */
    if (outfn != NULL) {
        write_kd_trees(outfn, root, n_feat);
    }

    ret = 0;

cleanup:
    if (root != NULL) {
        for (i = 0; i < n_feat; ++i) {
            if (root[i] != NULL)
                free_kd_tree(root[i]);
        }
        ckd_free(root);
    }
    if (r_veclen != NULL)
        ckd_free(r_veclen);
    if (veclen != NULL)
        ckd_free(veclen);
    if (means != NULL)
        ckd_free_4d(means);
    if (variances != NULL)
        ckd_free_4d(variances);

    return ret;
}
