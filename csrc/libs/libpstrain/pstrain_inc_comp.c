/**
 * @file pstrain_inc_comp.c
 * @brief Gaussian splitting (increase components) API for CFFI
 */

#include "pstrain_inc_comp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>

#include <s3/s3gau_io.h>
#include <s3/s3mixw_io.h>
#include <s3/gauden.h>
#include <s3/vector.h>
#include <s3/s3.h>

/**
 * Check if density k has already been split in this round.
 */
static int
not_done(uint32 k, uint32 *did, uint32 n_did)
{
    uint32 i;
    for (i = 0; (i < n_did) && (did[i] != k); i++);
    return (i >= n_did);
}

/**
 * Core density splitting algorithm.
 * Adapted from SphinxTrain's inc_densities.c
 */
static int
inc_densities_core(float32 ***new_mixw,
                   vector_t ***new_mean,
                   vector_t ***new_var,
                   float32 ***mixw,
                   vector_t ***mean,
                   vector_t ***var,
                   float32 ***dnom,
                   uint32 n_mixw,
                   uint32 n_mgau,
                   uint32 n_feat,
                   uint32 n_density,
                   const uint32 *veclen,
                   uint32 n_inc)
{
    uint32 i, j, k, l, r;
    uint32 **did;
    float32 max_wt;
    uint32 max_wt_idx;
    float32 std;

    assert(n_mgau <= n_mixw);

    if (n_mgau < n_mixw) {
        E_ERROR("Splitting of tied mixture gaussians not implemented\n");
        return -1;
    }

    /* Copy old parameters into new arrays */
    for (i = 0; i < n_mgau; i++) {
        for (j = 0; j < n_feat; j++) {
            for (k = 0; k < n_density; k++) {
                memcpy(new_mean[i][j][k], mean[i][j][k], veclen[j] * sizeof(float32));
                memcpy(new_var[i][j][k], var[i][j][k], veclen[j] * sizeof(float32));
                new_mixw[i][j][k] = mixw[i][j][k];
            }
        }
    }

    /* Track which densities have been split */
    did = (uint32 **)ckd_calloc_2d(n_feat, n_inc, sizeof(uint32));

    for (i = 0; i < n_mgau; i++) {
        if (i % 100 == 0) {
            E_INFO("Processing mixture %u/%u\n", i, n_mgau);
        }

        for (r = 0; r < n_inc; r++) {
            for (j = 0; j < n_feat; j++) {
                did[j][r] = n_density;

                /* Find density with largest count not yet split */
                max_wt = -1.0f;
                max_wt_idx = n_density;
                for (k = 0; k < n_density; k++) {
                    if ((max_wt < dnom[i][j][k]) && not_done(k, did[j], r)) {
                        max_wt = dnom[i][j][k];
                        max_wt_idx = k;
                    }
                }

                if (dnom[i][j][max_wt_idx] < 1e-38f) {
                    /* Never observed - copy from first density */
                    E_WARN("(mgau=%u, feat=%u, density=%u) never observed, skipping\n",
                           i, j, max_wt_idx);

                    new_mixw[i][j][n_density + r] = 0;
                    memcpy(new_var[i][j][n_density + r], var[i][j][0],
                           veclen[j] * sizeof(float32));
                    memcpy(new_mean[i][j][n_density + r], mean[i][j][0],
                           veclen[j] * sizeof(float32));
                    continue;
                }

                /* Split: mixw_a = mixw_b = mixw/2 */
                new_mixw[i][j][max_wt_idx] /= 2;
                new_mixw[i][j][n_density + r] = new_mixw[i][j][max_wt_idx];

                /* Variance unchanged */
                memcpy(new_var[i][j][n_density + r], var[i][j][max_wt_idx],
                       veclen[j] * sizeof(float32));

                /* mean_a = mean + 0.2*std, mean_b = mean - 0.2*std */
                for (l = 0; l < veclen[j]; l++) {
                    std = (float32)sqrt(var[i][j][max_wt_idx][l]);
                    new_mean[i][j][max_wt_idx][l] = mean[i][j][max_wt_idx][l] + 0.2f * std;
                    new_mean[i][j][n_density + r][l] = mean[i][j][max_wt_idx][l] - 0.2f * std;
                }

                did[j][r] = max_wt_idx;
            }
        }
    }

    ckd_free_2d((void **)did);
    return 0;
}

int
pstrain_inc_comp(const char *in_mean_path,
             const char *in_var_path,
             const char *in_mixw_path,
             const char *dcount_path,
             const char *out_mean_path,
             const char *out_var_path,
             const char *out_mixw_path,
             uint32 n_inc)
{
    vector_t ***mean = NULL;
    vector_t ***var = NULL;
    vector_t ***new_mean = NULL;
    vector_t ***new_var = NULL;
    float32 ***mixw = NULL;
    float32 ***new_mixw = NULL;
    float32 ***dnom = NULL;

    uint32 n_mixw, n_mgau, n_dnom;
    uint32 n_feat, n_density;
    uint32 *veclen = NULL;

    int ret = -1;

    /* Read mixture weights */
    E_INFO("Reading mixture weights from %s\n", in_mixw_path);
    if (s3mixw_read(in_mixw_path, &mixw, &n_mixw, &n_feat, &n_density) != S3_SUCCESS) {
        E_ERROR("Failed to read mixture weights\n");
        goto cleanup;
    }

    /* Validate n_inc */
    if (n_inc > n_density) {
        E_WARN("n_inc (%u) > n_density (%u), clamping to %u\n",
               n_inc, n_density, n_density);
        n_inc = n_density;
    }

    /* Read means */
    E_INFO("Reading means from %s\n", in_mean_path);
    if (s3gau_read(in_mean_path, &mean, &n_mgau, &n_feat, &n_density, &veclen) != S3_SUCCESS) {
        E_ERROR("Failed to read means\n");
        goto cleanup;
    }

    /* Read variances */
    E_INFO("Reading variances from %s\n", in_var_path);
    if (s3gau_read(in_var_path, &var, &n_mgau, &n_feat, &n_density, &veclen) != S3_SUCCESS) {
        E_ERROR("Failed to read variances\n");
        goto cleanup;
    }

    /* Read density counts */
    E_INFO("Reading density counts from %s\n", dcount_path);
    if (s3gaudnom_read(dcount_path, &dnom, &n_dnom, &n_feat, &n_density) != S3_SUCCESS) {
        E_ERROR("Failed to read density counts\n");
        goto cleanup;
    }

    E_INFO("Input: n_mgau=%u, n_feat=%u, n_density=%u, n_inc=%u\n",
           n_mgau, n_feat, n_density, n_inc);
    E_INFO("Output: n_density=%u\n", n_density + n_inc);

    /* Allocate output arrays */
    new_mean = gauden_alloc_param(n_mgau, n_feat, n_density + n_inc, veclen);
    new_var = gauden_alloc_param(n_mgau, n_feat, n_density + n_inc, veclen);
    new_mixw = (float32 ***)ckd_calloc_3d(n_mixw, n_feat, n_density + n_inc,
                                           sizeof(float32));

    /* Do the splitting */
    if (inc_densities_core(new_mixw, new_mean, new_var,
                           mixw, mean, var, dnom,
                           n_mixw, n_mgau, n_feat, n_density,
                           veclen, n_inc) != 0) {
        E_ERROR("Density splitting failed\n");
        goto cleanup;
    }

    /* Write outputs */
    E_INFO("Writing mixture weights to %s\n", out_mixw_path);
    if (s3mixw_write(out_mixw_path, new_mixw, n_mixw, n_feat,
                     n_density + n_inc) != S3_SUCCESS) {
        E_ERROR("Failed to write mixture weights\n");
        goto cleanup;
    }

    E_INFO("Writing means to %s\n", out_mean_path);
    if (s3gau_write(out_mean_path, (const vector_t ***)new_mean,
                    n_mgau, n_feat, n_density + n_inc, veclen) != S3_SUCCESS) {
        E_ERROR("Failed to write means\n");
        goto cleanup;
    }

    E_INFO("Writing variances to %s\n", out_var_path);
    if (s3gau_write(out_var_path, (const vector_t ***)new_var,
                    n_mgau, n_feat, n_density + n_inc, veclen) != S3_SUCCESS) {
        E_ERROR("Failed to write variances\n");
        goto cleanup;
    }

    E_INFO("Successfully split %u -> %u densities\n", n_density, n_density + n_inc);
    ret = 0;

cleanup:
    if (mean) gauden_free_param(mean);
    if (var) gauden_free_param(var);
    if (new_mean) gauden_free_param(new_mean);
    if (new_var) gauden_free_param(new_var);
    if (mixw) ckd_free_3d((void ***)mixw);
    if (new_mixw) ckd_free_3d((void ***)new_mixw);
    if (dnom) ckd_free_3d((void ***)dnom);
    if (veclen) ckd_free(veclen);

    return ret;
}
