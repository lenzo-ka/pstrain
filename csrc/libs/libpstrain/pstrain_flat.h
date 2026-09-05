/**
 * @file pstrain_flat.h
 * @brief Simplified flat initialization API for CFFI
 */

#ifndef PSTRAIN_FLAT_H
#define PSTRAIN_FLAT_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create flat transition matrices from topology file
 *
 * @param mdef_path Path to model definition file
 * @param topo_path Path to topology file
 * @param output_path Output path for transition matrices
 * @return 0 on success, -1 on error
 */
int
pstrain_flat_tmat(const char *mdef_path,
              const char *topo_path,
              const char *output_path);

/**
 * Create flat mixture weights (uniform)
 *
 * @param n_tied_state Number of tied states
 * @param n_stream Number of feature streams (usually 1)
 * @param n_density Number of densities per mixture
 * @param output_path Output path for mixture weights
 * @return 0 on success, -1 on error
 */
int
pstrain_flat_mixw(uint32 n_tied_state,
              uint32 n_stream,
              uint32 n_density,
              const char *output_path);

/**
 * Initialize Gaussian parameters from feature data
 *
 * Computes mean (first pass) or variance (second pass) from features.
 * For global mode (mdef_path=NULL), computes single global statistics.
 * For per-state mode, requires segmentation files.
 *
 * @param mdef_path Path to model definition (NULL for global mode)
 * @param dict_path Path to dictionary (NULL for global mode)
 * @param filler_dict_path Path to filler dictionary (may be NULL)
 * @param feat_type Feature type (e.g., "1s_c_d_dd")
 * @param ceplen Cepstral length (e.g., 13)
 * @param ctl_path Path to control file (list of utterances)
 * @param cep_dir Directory containing feature files
 * @param cep_ext Feature file extension (e.g., ".mfc")
 * @param lsn_path Path to transcription file (may be NULL for global)
 * @param seg_dir Directory containing segmentation files (may be NULL)
 * @param seg_ext Segmentation file extension (may be NULL)
 * @param accum_dir Directory to write accumulator files
 * @param mean_path Path to means file for variance pass (NULL for mean pass)
 * @return the number of utterances skipped (0 when all were used), or -1 on
 *         error
 */
int
pstrain_init_gau(const char *mdef_path,
             const char *dict_path,
             const char *filler_dict_path,
             const char *feat_type,
             int32 ceplen,
             const char *ctl_path,
             const char *cep_dir,
             const char *cep_ext,
             const char *lsn_path,
             const char *seg_dir,
             const char *seg_ext,
             const char *accum_dir,
             const char *mean_path);

/**
 * Normalize accumulated counts to get model parameters
 *
 * @param accum_dir Directory containing accumulator files
 * @param mean_path Output path for means (NULL to skip)
 * @param var_path Output path for variances (NULL to skip)
 * @return 0 on success, -1 on error
 */
int
pstrain_norm_gau(const char *accum_dir,
             const char *mean_path,
             const char *var_path);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_FLAT_H */
