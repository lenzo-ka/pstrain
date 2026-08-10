#ifndef PSTRAIN_DELINT_H
#define PSTRAIN_DELINT_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform deleted interpolation to smooth mixture weights.
 *
 * Deleted interpolation smooths CD mixture weights by interpolating
 * between CD, CI, and uniform distributions using held-out data.
 *
 * @param moddeffn Model definition file
 * @param mixwfn Output mixture weight file
 * @param accumdirs NULL-terminated array of accumulator directories
 * @param cilambda CI interpolation weight (0.0-1.0, default 0.9)
 * @param maxiter Max iterations for convergence (default 100)
 * @return 0 on success, non-zero on error
 */
int pstrain_delint(const char *moddeffn,
               const char *mixwfn,
               const char **accumdirs,
               float32 cilambda,
               int32 maxiter);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_DELINT_H */
