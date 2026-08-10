/**
 * @file pstrain_inc_comp.h
 * @brief Gaussian splitting (increase components) API for CFFI
 *
 * Wraps SphinxTrain's inc_comp functionality for increasing
 * the number of Gaussian components per mixture.
 */

#ifndef PSTRAIN_INC_COMP_H
#define PSTRAIN_INC_COMP_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Increase the number of Gaussian components by splitting.
 *
 * Splits the most probable Gaussians (by count) into two,
 * perturbing means by +/- 0.2 * std.
 *
 * @param in_mean_path Input means file
 * @param in_var_path Input variances file
 * @param in_mixw_path Input mixture weights file
 * @param dcount_path Density counts file (from BW norm)
 * @param out_mean_path Output means file
 * @param out_var_path Output variances file
 * @param out_mixw_path Output mixture weights file
 * @param n_inc Number of components to add (typically doubles: n_inc = n_density)
 * @return 0 on success, -1 on error
 */
int
pstrain_inc_comp(const char *in_mean_path,
             const char *in_var_path,
             const char *in_mixw_path,
             const char *dcount_path,
             const char *out_mean_path,
             const char *out_var_path,
             const char *out_mixw_path,
             uint32 n_inc);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_INC_COMP_H */
