#include "pstrain_map_adapt.h"

#include <s3/common.h>
#include <s3/model_inventory.h>
#include <s3/model_def_io.h>
#include <s3/s3gau_io.h>
#include <s3/s3mixw_io.h>
#include <s3/s3tmat_io.h>
#include <s3/s3acc_io.h>
#include <s3/s3.h>
#include <s3/ts2cb.h>
#include <s3/s3ts2cb_io.h>
#include <s3/gauden.h>
#include <sphinxbase/matrix.h>
#include <sphinxbase/err.h>

#include <stdio.h>
#include <string.h>

/* Forward declarations from map_adapt/main.c */
extern void check_consistency(const char *filename,
                              uint32 n_mgau, uint32 n_mgau_rd,
                              uint32 n_stream, uint32 n_stream_rd,
                              uint32 n_density, uint32 n_density_rd,
                              const uint32 *veclen, const uint32 *veclen_rd);

extern float32 *** estimate_tau(vector_t ***si_mean, vector_t ***si_var, float32 ***si_mixw,
                                uint32 n_cb, uint32 n_stream, uint32 n_density, uint32 n_mixw,
                                const uint32 *veclen, vector_t ***wt_mean, float32 ***wt_mixw,
                                float32 ***wt_dcount);

extern int map_mixw_reest(model_def_t *mdef, float32 ***map_tau, float32 fixed_tau,
                          float32 ***si_mixw, float32 ***wt_mixw, float32 ***map_mixw,
                          float32 mwfloor, uint32 n_cb, uint32 n_mixw, uint32 n_stream, uint32 n_density);

extern int map_tmat_reest(float32 ***si_tmat, float32 ***wt_tmat, float32 ***map_tmat,
                          float32 tpfloor, uint32 n_tmat, uint32 n_state);

extern int32 bayes_mean_reest(vector_t ***si_mean, vector_t ***si_var,
                              vector_t ***wt_mean, vector_t ***wt_var,
                              float32 ***wt_dcount, int32 pass2var,
                              vector_t ***map_mean, float32 varfloor,
                              uint32 i, uint32 j, uint32 k, const uint32 *veclen);

extern int map_mean_reest(float32 tau, vector_t ***si_mean, vector_t ***wt_mean,
                          float32 ***wt_dcount, vector_t ***map_mean,
                          uint32 i, uint32 j, uint32 k, const uint32 *veclen);

extern int map_var_reest(float32 tau, vector_t ***si_mean, vector_t ***si_var,
                         vector_t ***wt_mean, vector_t ***wt_var, float32 ***wt_dcount,
                         vector_t ***map_mean, vector_t ***map_var, float32 varfloor,
                         uint32 i, uint32 j, uint32 k, const uint32 *veclen);

int pstrain_map_adapt(const char *meanfn,
                  const char *varfn,
                  const char *mixwfn,
                  const char *tmatfn,
                  const char **accumdirs,
                  const char *mapmeanfn,
                  const char *mapvarfn,
                  const char *mapmixwfn,
                  const char *maptmatfn,
                  const char *moddeffn,
                  const char *ts2cbfn,
                  float32 tau,
                  int32 use_fixed_tau,
                  int32 use_bayes_mean,
                  float32 mwfloor,
                  float32 varfloor,
                  float32 tpfloor)
{
    float32 ***si_mixw = NULL;
    float32 ***si_tmat = NULL;
    vector_t ***si_mean = NULL;
    vector_t ***si_var = NULL;

    vector_t ***wt_mean = NULL;
    vector_t ***wt_var = NULL;
    float32 ***wt_mixw = NULL;
    float32 ***wt_tmat = NULL;
    float32 ***wt_dcount = NULL;
    int32 pass2var;

    float32 ***map_mixw = NULL;
    float32 ***map_tmat = NULL;
    vector_t ***map_mean = NULL;
    vector_t ***map_var = NULL;
    float32 ***map_tau = NULL;

    uint32 n_mixw = 0, n_mixw_rd;
    uint32 n_tmat = 0, n_tmat_rd, n_state = 0, n_state_rd;
    uint32 n_cb, n_cb_rd;
    uint32 n_stream, n_stream_rd;
    uint32 n_density, n_density_rd;
    uint32 *veclen = NULL;
    uint32 *veclen_rd = NULL;

    model_def_t *mdef = NULL;
    uint32 i, j, k;
    int ret = -1;

    /* Validate required arguments */
    if (meanfn == NULL || varfn == NULL) {
        E_ERROR("Must specify baseline means and variances\n");
        return -1;
    }
    if (mapmeanfn == NULL) {
        E_ERROR("Must specify output MAP means\n");
        return -1;
    }
    if (accumdirs == NULL || accumdirs[0] == NULL) {
        E_ERROR("Must specify at least one accumulator directory\n");
        return -1;
    }

    /* Read SI model parameters */
    if (s3gau_read(meanfn, &si_mean, &n_cb, &n_stream, &n_density, &veclen) != S3_SUCCESS) {
        E_ERROR("Couldn't read %s\n", meanfn);
        return -1;
    }
    if (s3gau_read(varfn, &si_var, &n_cb_rd, &n_stream_rd, &n_density_rd, &veclen_rd) != S3_SUCCESS) {
        E_ERROR("Couldn't read %s\n", varfn);
        goto cleanup;
    }
    check_consistency(varfn, n_cb, n_cb_rd, n_stream, n_stream_rd,
                      n_density, n_density_rd, veclen, veclen_rd);

    /* Read and normalize SI mixture weights */
    if (mixwfn) {
        if (s3mixw_read(mixwfn, &si_mixw, &n_mixw, &n_stream_rd, &n_density_rd) != S3_SUCCESS) {
            E_ERROR("Couldn't read %s\n", mixwfn);
            goto cleanup;
        }
        for (i = 0; i < n_mixw; ++i) {
            for (j = 0; j < n_stream; ++j) {
                float32 sum_si_mixw = 0.0f;
                for (k = 0; k < n_density; ++k) {
                    if (si_mixw[i][j][k] < mwfloor)
                        si_mixw[i][j][k] = mwfloor;
                    sum_si_mixw += si_mixw[i][j][k];
                }
                for (k = 0; k < n_density; ++k)
                    si_mixw[i][j][k] /= sum_si_mixw;
            }
        }
    }

    /* Read SI transition matrices */
    if (tmatfn) {
        if (s3tmat_read(tmatfn, &si_tmat, &n_tmat, &n_state) != S3_SUCCESS) {
            E_ERROR("Couldn't read %s\n", tmatfn);
            goto cleanup;
        }
    }

    /* Read observation counts from accumulator directories */
    for (i = 0; accumdirs[i]; ++i) {
        E_INFO("Reading and accumulating observation counts from %s\n", accumdirs[i]);

        if (rdacc_den(accumdirs[i], &wt_mean, &wt_var, &pass2var, &wt_dcount,
                      &n_cb_rd, &n_stream_rd, &n_density_rd, &veclen_rd) != S3_SUCCESS) {
            E_ERROR("Error reading densities from %s\n", accumdirs[i]);
            goto cleanup;
        }
        check_consistency(accumdirs[i], n_cb, n_cb_rd, n_stream, n_stream_rd,
                          n_density, n_density_rd, veclen, veclen_rd);

        if (pass2var && mapvarfn) {
            E_ERROR("Variance re-estimation requested, but -2passvar was specified in bw.\n");
            goto cleanup;
        }

        if (mapmixwfn || !use_fixed_tau) {
            if (rdacc_mixw(accumdirs[i], &wt_mixw, &n_mixw_rd, &n_stream_rd, &n_density_rd) != S3_SUCCESS) {
                E_ERROR("Error reading mixture weights from %s\n", accumdirs[i]);
                goto cleanup;
            }
            check_consistency(accumdirs[i], n_mixw, n_mixw_rd, n_stream, n_stream_rd,
                              n_density, n_density_rd, veclen, veclen_rd);
        }

        if (maptmatfn) {
            if (rdacc_tmat(accumdirs[i], &wt_tmat, &n_tmat_rd, &n_state_rd) != S3_SUCCESS) {
                E_ERROR("Error reading transition matrices from %s\n", accumdirs[i]);
                goto cleanup;
            }
            if (n_tmat_rd != n_tmat || n_state_rd != n_state) {
                E_ERROR("Mismatch in transition matrices from %s\n", accumdirs[i]);
                goto cleanup;
            }
        }
    }

    if (veclen_rd) {
        ckd_free(veclen_rd);
        veclen_rd = NULL;
    }

    /* Allocate MAP parameters */
    map_mean = gauden_alloc_param(n_cb, n_stream, n_density, veclen);
    if (mapvarfn)
        map_var = gauden_alloc_param(n_cb, n_stream, n_density, veclen);
    if (mapmixwfn)
        map_mixw = (float32 ***)ckd_calloc_3d(n_mixw, n_stream, n_density, sizeof(float32));
    if (maptmatfn)
        map_tmat = (float32 ***)ckd_calloc_3d(n_tmat, n_state-1, n_state, sizeof(float32));

    /* Estimate or use fixed tau */
    if (use_fixed_tau) {
        E_INFO("tau hyperparameter fixed at %f\n", tau);
    } else {
        map_tau = estimate_tau(si_mean, si_var, si_mixw,
                               n_cb, n_stream, n_density, n_mixw, veclen,
                               wt_mean, wt_mixw, wt_dcount);
    }

    /* Re-estimate mixture weights */
    if (map_mixw) {
        if (moddeffn) {
            E_INFO("Reading %s\n", moddeffn);
            if (model_def_read(&mdef, moddeffn) != S3_SUCCESS) {
                E_ERROR("Couldn't read %s\n", moddeffn);
                goto cleanup;
            }

            if (ts2cbfn) {
                if (strcmp(SEMI_LABEL, ts2cbfn) == 0) {
                    mdef->cb = semi_ts2cb(mdef->n_tied_state);
                } else if (strcmp(CONT_LABEL, ts2cbfn) == 0) {
                    mdef->cb = cont_ts2cb(mdef->n_tied_state);
                } else if (strcmp(PTM_LABEL, ts2cbfn) == 0) {
                    mdef->cb = ptm_ts2cb(mdef);
                } else if (s3ts2cb_read(ts2cbfn, &mdef->cb, NULL, NULL) != S3_SUCCESS) {
                    E_ERROR("Couldn't read %s\n", ts2cbfn);
                    goto cleanup;
                }
            }
        }

        if (map_mixw_reest(mdef, map_tau, tau, si_mixw, wt_mixw, map_mixw,
                           mwfloor, n_cb, n_mixw, n_stream, n_density) != S3_SUCCESS) {
            E_ERROR("Mixture weight re-estimation failed\n");
            goto cleanup;
        }
    }

    /* Re-estimate transition matrices */
    if (map_tmat) {
        if (map_tmat_reest(si_tmat, wt_tmat, map_tmat, tpfloor, n_tmat, n_state) != S3_SUCCESS) {
            E_ERROR("Transition matrix re-estimation failed\n");
            goto cleanup;
        }
    }

    /* Re-estimate means and variances */
    if (use_bayes_mean)
        E_INFO("Re-estimating means using Bayesian interpolation\n");
    else
        E_INFO("Re-estimating means using MAP\n");
    if (map_var)
        E_INFO("Re-estimating variances using MAP\n");

    for (i = 0; i < n_cb; ++i) {
        for (j = 0; j < n_stream; ++j) {
            for (k = 0; k < n_density; ++k) {
                float32 cur_tau = (map_tau == NULL) ? tau : map_tau[i][j][k];

                if (use_bayes_mean) {
                    bayes_mean_reest(si_mean, si_var, wt_mean, wt_var,
                                     wt_dcount, pass2var, map_mean, varfloor,
                                     i, j, k, veclen);
                } else {
                    map_mean_reest(cur_tau, si_mean, wt_mean, wt_dcount,
                                   map_mean, i, j, k, veclen);
                }

                if (map_var) {
                    map_var_reest(cur_tau, si_mean, si_var, wt_mean, wt_var,
                                  wt_dcount, map_mean, map_var, varfloor,
                                  i, j, k, veclen);
                }
            }
        }
    }

    /* Write output files */
    if (mapmeanfn) {
        if (s3gau_write(mapmeanfn, (const vector_t ***)map_mean,
                        n_cb, n_stream, n_density, veclen) != S3_SUCCESS) {
            E_ERROR("Unable to write MAP mean to %s\n", mapmeanfn);
            goto cleanup;
        }
    }

    if (map_var && mapvarfn) {
        if (s3gau_write(mapvarfn, (const vector_t ***)map_var,
                        n_cb, n_stream, n_density, veclen) != S3_SUCCESS) {
            E_ERROR("Unable to write MAP variance to %s\n", mapvarfn);
            goto cleanup;
        }
    }

    if (map_mixw && mapmixwfn) {
        if (s3mixw_write(mapmixwfn, map_mixw, n_mixw, n_stream, n_density) != S3_SUCCESS) {
            E_ERROR("Unable to write MAP mixture weights to %s\n", mapmixwfn);
            goto cleanup;
        }
    }

    if (map_tmat && maptmatfn) {
        if (s3tmat_write(maptmatfn, map_tmat, n_tmat, n_state) != S3_SUCCESS) {
            E_ERROR("Unable to write MAP transition matrices to %s\n", maptmatfn);
            goto cleanup;
        }
    }

    ret = 0;

cleanup:
    if (mdef)
        model_def_free(mdef);
    if (veclen)
        ckd_free(veclen);
    if (veclen_rd)
        ckd_free(veclen_rd);
    if (si_mean)
        gauden_free_param(si_mean);
    if (si_var)
        gauden_free_param(si_var);
    if (si_mixw)
        ckd_free_3d(si_mixw);
    if (si_tmat)
        ckd_free_3d(si_tmat);
    if (wt_mean)
        gauden_free_param(wt_mean);
    if (wt_var)
        gauden_free_param(wt_var);
    if (wt_dcount)
        ckd_free_3d(wt_dcount);
    if (wt_mixw)
        ckd_free_3d(wt_mixw);
    if (wt_tmat)
        ckd_free_3d(wt_tmat);
    if (map_mean)
        gauden_free_param(map_mean);
    if (map_var)
        gauden_free_param(map_var);
    if (map_tau)
        ckd_free_3d(map_tau);
    if (map_mixw)
        ckd_free_3d(map_mixw);
    if (map_tmat)
        ckd_free_3d(map_tmat);

    return ret;
}
/* ====================================================================
 * Copyright (c) 1999-2016 Carnegie Mellon University.  All rights
 * reserved.
 *
 * This file is derived from CMU SphinxTrain/SphinxBase sources.
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
