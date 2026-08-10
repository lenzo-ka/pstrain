/**
 * @file pstrain_agg_seg.h
 * @brief CFFI-friendly wrapper for segment aggregation.
 *
 * Aggregates feature observations by state, phone, or all together.
 */

#ifndef PSTRAIN_AGG_SEG_H
#define PSTRAIN_AGG_SEG_H

#include <sphinxbase/prim_type.h>

/**
 * Segment aggregation type.
 */
typedef enum {
    PSTRAIN_SEGTYPE_ALL = 0,  /**< All frames to one file */
    PSTRAIN_SEGTYPE_ST = 1,   /**< Aggregate by tied state */
    PSTRAIN_SEGTYPE_PHN = 2   /**< Aggregate by phone */
} pstrain_segtype_t;

/**
 * Aggregate feature segments from training corpus.
 *
 * This function aggregates feature vectors from a training corpus,
 * grouping them by state, phone, or writing all to one file.
 *
 * @param mdef_path Model definition file
 * @param dict_path Dictionary file
 * @param fdict_path Filler dictionary file (optional)
 * @param ctl_path Control file listing utterances
 * @param cep_dir Cepstrum directory
 * @param cep_ext Cepstrum extension
 * @param seg_dir Segmentation directory (optional, for st/phn modes)
 * @param seg_ext Segmentation extension
 * @param output_path Output dump file path
 * @param index_path Index file path (optional, for st/phn modes)
 * @param ts2cb_path Tied-state to codebook mapping (".semi.", ".cont.", or file)
 * @param cnt_path Count file path (for st/phn modes, created if not exists)
 * @param segtype Segment type (0=all, 1=st, 2=phn)
 * @param feat_type Feature type string (e.g., "1s_c_d_dd")
 * @param ceplen Cepstrum length
 * @param stride Take every stride-th frame (default 1)
 * @param cachesz Cache size in MB (default 200)
 * @return 0 on success, -1 on error
 */
int pstrain_agg_seg(const char *mdef_path,
                const char *dict_path,
                const char *fdict_path,
                const char *ctl_path,
                const char *cep_dir,
                const char *cep_ext,
                const char *seg_dir,
                const char *seg_ext,
                const char *output_path,
                const char *index_path,
                const char *ts2cb_path,
                const char *cnt_path,
                int32 segtype,
                const char *feat_type,
                int32 ceplen,
                int32 stride,
                int32 cachesz);

#endif /* PSTRAIN_AGG_SEG_H */
