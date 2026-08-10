/**
 * @file pstrain_kmeans.h
 * @brief K-means clustering API for CFFI
 *
 * Wraps SphinxTrain's k-means with a simple array-based interface.
 */

#ifndef PSTRAIN_KMEANS_H
#define PSTRAIN_KMEANS_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run k-means clustering on observation data.
 *
 * @param observations Flat array of observations [n_obs * veclen]
 * @param n_obs Number of observations
 * @param veclen Vector length (dimension)
 * @param n_clusters Number of clusters (k)
 * @param max_iter Maximum iterations
 * @param min_ratio Convergence ratio threshold
 * @param out_centroids Output centroids [n_clusters * veclen] (pre-allocated)
 * @param out_labels Output cluster assignments [n_obs] (pre-allocated, or NULL)
 * @return Squared error on success, negative on error
 */
float64
pstrain_kmeans(const float32 *observations,
           uint32 n_obs,
           uint32 veclen,
           uint32 n_clusters,
           uint32 max_iter,
           float32 min_ratio,
           float32 *out_centroids,
           uint32 *out_labels);

/**
 * Initialize Gaussian means for a tied state using k-means on features.
 *
 * This is used after inc_comp to re-cluster the means for better initialization.
 *
 * @param features Flat feature array [n_frames * veclen]
 * @param n_frames Number of feature frames
 * @param veclen Feature vector length
 * @param n_density Target number of Gaussians
 * @param max_iter Maximum k-means iterations
 * @param min_ratio Convergence threshold
 * @param out_means Output means [n_density * veclen] (pre-allocated)
 * @param out_vars Output variances [n_density * veclen] (pre-allocated, or NULL for no variance)
 * @param out_weights Output mixture weights [n_density] (pre-allocated, or NULL)
 * @return 0 on success, -1 on error
 */
int
pstrain_kmeans_init(const float32 *features,
                uint32 n_frames,
                uint32 veclen,
                uint32 n_density,
                uint32 max_iter,
                float32 min_ratio,
                float32 *out_means,
                float32 *out_vars,
                float32 *out_weights);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_KMEANS_H */
