#ifndef PSTRAIN_KDTREE_H
#define PSTRAIN_KDTREE_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Build KD-trees for fast Gaussian selection.
 *
 * @param meanfn Input means file (S3 format)
 * @param varfn Input variances file (S3 format)
 * @param outfn Output KD-trees file (NULL to skip writing)
 * @param threshold Gaussian box threshold (0.0-1.0)
 * @param depth Number of tree levels
 * @param absolute Use absolute threshold calculation
 * @return 0 on success, non-zero on error
 */
int pstrain_kdtree_build(const char *meanfn,
                     const char *varfn,
                     const char *outfn,
                     float32 threshold,
                     int32 depth,
                     int32 absolute);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_KDTREE_H */
