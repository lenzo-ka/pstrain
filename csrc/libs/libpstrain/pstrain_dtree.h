/**
 * @file pstrain_dtree.h
 * @brief CFFI-friendly wrappers for decision tree operations.
 *
 * Provides simplified API for building decision trees and generating questions.
 */

#ifndef PSTRAIN_DTREE_H
#define PSTRAIN_DTREE_H

#include <sphinxbase/prim_type.h>
#include <s3/quest.h>
#include <s3/pset_io.h>
#include <s3/dtree.h>
#include <s3/acmod_set.h>

/**
 * Read a phone set (question) file.
 *
 * @param filename Path to the phone set file
 * @param mdef_path Path to model definition file (for phone names)
 * @param out_n_pset Output: number of phone sets read
 * @return pset_t array, or NULL on error
 */
pset_t *pstrain_read_pset(const char *filename,
                       const char *mdef_path,
                       uint32 *out_n_pset);

/**
 * Free a phone set array.
 */
void pstrain_free_pset(pset_t *pset, uint32 n_pset);

/**
 * Build a decision tree for triphones of a given base phone.
 *
 * This is the main entry point for building decision trees. It:
 * 1. Loads model data (mdef, mixw, means, vars)
 * 2. Generates questions from the phone set
 * 3. Builds the decision tree
 * 4. Writes the tree to the output file
 *
 * @param mdef_path Model definition file (with triphones)
 * @param mixw_path Mixture weights file
 * @param mean_path Means file (for continuous models)
 * @param var_path Variance file (for continuous models)
 * @param pset_path Phone set (question) file
 * @param output_path Output tree file path
 * @param phone Base phone name (e.g., "AA")
 * @param state State index (0-based)
 * @param continuous 1 for continuous, 0 for semi-continuous
 * @param ssplitmin Minimum simple split count
 * @param ssplitmax Maximum simple split count
 * @param ssplitthr Simple split threshold
 * @param csplitmin Minimum composite split count
 * @param csplitmax Maximum composite split count
 * @param csplitthr Composite split threshold
 * @param mwfloor Mixture weight floor
 * @param varfloor Variance floor
 * @param cntthresh Count threshold for model inclusion
 * @param stwt State weights (n_state floats, or NULL for uniform)
 * @param n_stwt Number of state weights
 * @param allphones Build for all phones at once
 * @return 0 on success, -1 on error
 */
int pstrain_build_tree(const char *mdef_path,
                   const char *mixw_path,
                   const char *mean_path,
                   const char *var_path,
                   const char *pset_path,
                   const char *output_path,
                   const char *phone,
                   uint32 state,
                   int32 continuous,
                   uint32 ssplitmin,
                   uint32 ssplitmax,
                   float32 ssplitthr,
                   uint32 csplitmin,
                   uint32 csplitmax,
                   float32 csplitthr,
                   float32 mwfloor,
                   float32 varfloor,
                   float32 cntthresh,
                   float32 *stwt,
                   uint32 n_stwt,
                   int32 allphones);

/**
 * Tie states using decision trees.
 *
 * @param input_mdef_path Input model definition file (untied)
 * @param output_mdef_path Output model definition file (tied)
 * @param tree_dir Directory containing decision tree files
 * @param pset_path Phone set file
 * @param phone Phone to tie (or NULL for all)
 * @param allphones Tie all phones
 * @return 0 on success, -1 on error
 */
int pstrain_tie_states(const char *input_mdef_path,
                   const char *output_mdef_path,
                   const char *tree_dir,
                   const char *pset_path,
                   const char *phone,
                   int32 allphones);

/**
 * Generate phonetic questions by clustering CI distributions.
 *
 * @param mdef_path CI model definition file
 * @param mixw_path Mixture weights file
 * @param mean_path Means file (for continuous)
 * @param var_path Variance file (for continuous)
 * @param output_path Output question file
 * @param continuous 1 for continuous, 0 for semi-continuous
 * @param npermute Number of permutations for clustering
 * @param quests_per_state Questions per state
 * @param varfloor Variance floor
 * @param niter Number of iterations
 * @return 0 on success, -1 on error
 */
int pstrain_make_quests(const char *mdef_path,
                    const char *mixw_path,
                    const char *mean_path,
                    const char *var_path,
                    const char *output_path,
                    int32 continuous,
                    uint32 npermute,
                    uint32 quests_per_state,
                    float32 varfloor,
                    uint32 niter);

/**
 * Prune decision trees to a target number of senones.
 *
 * Removes bifurcations that resulted in minimum likelihood increase,
 * pruning globally across all decision trees.
 *
 * @param mdef_path CI model definition file
 * @param pset_path Phone set (question) file
 * @param input_tree_dir Input tree directory
 * @param output_tree_dir Output tree directory
 * @param n_seno_target Target number of senones
 * @param min_occ Prune nodes with fewer than this many observations
 * @param allphones Prune all phones together as single tree
 * @return 0 on success, -1 on error
 */
int pstrain_prune_tree(const char *mdef_path,
                   const char *pset_path,
                   const char *input_tree_dir,
                   const char *output_tree_dir,
                   uint32 n_seno_target,
                   float32 min_occ,
                   int32 allphones);

/**
 * Initialize tied CD model parameters from a CI model.
 *
 * Maps CI phone parameters to CD triphone tied states based on the
 * model definitions. For each triphone in the destination mdef:
 * - If an exact match exists in source, copy its parameters
 * - If only base phone exists, use base phone parameters
 * - Otherwise, initialize with uniform distribution
 *
 * @param src_mdef_path Source (CI) model definition file
 * @param src_mixw_path Source mixture weights file
 * @param src_mean_path Source means file
 * @param src_var_path Source variances file
 * @param src_tmat_path Source transition matrices file
 * @param dest_mdef_path Destination (CD tied) model definition file
 * @param dest_mixw_path Output mixture weights file
 * @param dest_mean_path Output means file
 * @param dest_var_path Output variances file
 * @param dest_tmat_path Output transition matrices file
 * @param continuous 1 for continuous, 0 for semi-continuous
 * @return 0 on success, -1 on error
 */
int pstrain_init_mixw(const char *src_mdef_path,
                  const char *src_mixw_path,
                  const char *src_mean_path,
                  const char *src_var_path,
                  const char *src_tmat_path,
                  const char *dest_mdef_path,
                  const char *dest_mixw_path,
                  const char *dest_mean_path,
                  const char *dest_var_path,
                  const char *dest_tmat_path,
                  int32 continuous);

#endif /* PSTRAIN_DTREE_H */
