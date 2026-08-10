/**
 * @file pstrain_kmeans.c
 * @brief K-means clustering API for CFFI
 *
 * Provides array-based wrappers around SphinxTrain's k-means functions.
 */

#include "pstrain_kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>

#include <s3/kmeans.h>
#include <s3/vector.h>
#include <s3/s3.h>

/* Global state for the get_obs callback */
static const float32 *g_observations = NULL;
static uint32 g_veclen = 0;

void
pstrain_kmeans_session_reset(void)
{
    g_observations = NULL;
    g_veclen = 0;
}

/**
 * Callback for k_means to get observation i.
 */
static vector_t
pstrain_get_obs(uint32 i)
{
    return (vector_t)(g_observations + i * g_veclen);
}

float64
pstrain_kmeans(const float32 *observations,
           uint32 n_obs,
           uint32 veclen,
           uint32 n_clusters,
           uint32 max_iter,
           float32 min_ratio,
           float32 *out_centroids,
           uint32 *out_labels)
{
    vector_t *mean = NULL;
    codew_t *labels = NULL;
    float64 sqerr;
    uint32 i, j;

    if (!observations || !out_centroids || n_obs == 0 || veclen == 0) {
        E_ERROR("Invalid arguments to pstrain_kmeans\n");
        return -1.0;
    }

    /* Set up global state for callback */
    g_observations = observations;
    g_veclen = veclen;
    k_means_set_get_obs(&pstrain_get_obs);

    /* Allocate mean vectors */
    mean = (vector_t *)ckd_calloc_2d(n_clusters, veclen, sizeof(float32));
    if (!mean) {
        E_ERROR("Failed to allocate mean vectors\n");
        return -1.0;
    }

    /* Initialize means from random observations */
    srand48(0);
    for (i = 0; i < n_clusters; i++) {
        uint32 idx = (uint32)(drand48() * n_obs);
        if (idx >= n_obs) idx = n_obs - 1;
        memcpy(mean[i], observations + idx * veclen, veclen * sizeof(float32));
    }

    /* Run k-means */
    if (n_clusters > 1) {
        sqerr = k_means_trineq(mean, n_clusters, n_obs, veclen,
                               min_ratio, max_iter, &labels);
    } else {
        sqerr = k_means(mean, n_clusters, n_obs, veclen,
                        min_ratio, max_iter, &labels);
    }

    if (sqerr < 0) {
        E_ERROR("K-means failed\n");
        ckd_free_2d((void **)mean);
        if (labels) ckd_free(labels);
        return sqerr;
    }

    /* Copy results */
    for (i = 0; i < n_clusters; i++) {
        memcpy(out_centroids + i * veclen, mean[i], veclen * sizeof(float32));
    }

    if (out_labels && labels) {
        for (i = 0; i < n_obs; i++) {
            out_labels[i] = labels[i];
        }
    }

    ckd_free_2d((void **)mean);
    if (labels) ckd_free(labels);

    return sqerr;
}

int
pstrain_kmeans_init(const float32 *features,
                uint32 n_frames,
                uint32 veclen,
                uint32 n_density,
                uint32 max_iter,
                float32 min_ratio,
                float32 *out_means,
                float32 *out_vars,
                float32 *out_weights)
{
    uint32 *labels = NULL;
    uint32 *counts = NULL;
    float64 sqerr;
    uint32 i, j, k;
    float64 diff;

    if (!features || !out_means || n_frames == 0) {
        E_ERROR("Invalid arguments to pstrain_kmeans_init\n");
        return -1;
    }

    /* Allocate labels */
    labels = (uint32 *)ckd_calloc(n_frames, sizeof(uint32));
    if (!labels) {
        E_ERROR("Failed to allocate labels\n");
        return -1;
    }

    /* Run k-means to get cluster centers */
    sqerr = pstrain_kmeans(features, n_frames, veclen, n_density,
                       max_iter, min_ratio, out_means, labels);

    if (sqerr < 0) {
        ckd_free(labels);
        return -1;
    }

    /* Compute mixture weights from label counts */
    if (out_weights) {
        counts = (uint32 *)ckd_calloc(n_density, sizeof(uint32));
        for (i = 0; i < n_frames; i++) {
            counts[labels[i]]++;
        }
        for (k = 0; k < n_density; k++) {
            out_weights[k] = (float32)counts[k] / (float32)n_frames;
        }
        ckd_free(counts);
    }

    /* Compute variances from cluster assignments */
    if (out_vars) {
        counts = (uint32 *)ckd_calloc(n_density, sizeof(uint32));

        /* Zero out variances */
        memset(out_vars, 0, n_density * veclen * sizeof(float32));

        /* Accumulate squared differences */
        for (i = 0; i < n_frames; i++) {
            k = labels[i];
            counts[k]++;
            for (j = 0; j < veclen; j++) {
                diff = features[i * veclen + j] - out_means[k * veclen + j];
                out_vars[k * veclen + j] += (float32)(diff * diff);
            }
        }

        /* Normalize */
        for (k = 0; k < n_density; k++) {
            if (counts[k] > 0) {
                float32 norm = 1.0f / (float32)counts[k];
                for (j = 0; j < veclen; j++) {
                    out_vars[k * veclen + j] *= norm;
                    /* Floor variance */
                    if (out_vars[k * veclen + j] < 1e-4f) {
                        out_vars[k * veclen + j] = 1e-4f;
                    }
                }
            } else {
                /* Unobserved cluster - use unit variance */
                for (j = 0; j < veclen; j++) {
                    out_vars[k * veclen + j] = 1.0f;
                }
            }
        }

        ckd_free(counts);
    }

    ckd_free(labels);

    E_INFO("K-means init: n_frames=%u, n_density=%u, sqerr=%e\n",
           n_frames, n_density, sqerr);

    return 0;
}
