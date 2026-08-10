#ifndef PSTRAIN_PARAM_CNT_H
#define PSTRAIN_PARAM_CNT_H

#include <sphinxbase/prim_type.h>

/* Parameter count types */
#define PARAM_CNT_STATE 0
#define PARAM_CNT_CB    1
#define PARAM_CNT_PHONE 2

/**
 * @brief Count parameter occurrences in the training corpus.
 *
 * This function wraps the functionality of the SphinxTrain `param_cnt` program,
 * counting occurrences of states, codebooks, or phones in the training data.
 *
 * @param mdef_path Model definition file.
 * @param dict_path Dictionary file.
 * @param fdict_path Filler dictionary file (optional, can be NULL).
 * @param ctl_path Control file listing utterances.
 * @param lsn_path Transcript file.
 * @param ts2cb_path Tied-state to codebook mapping (optional, can be NULL, ".semi.", ".cont.").
 * @param seg_dir Segmentation directory (optional, can be NULL).
 * @param seg_ext Segmentation extension (default "v8_seg").
 * @param output_path Output file path (optional, NULL for stdout).
 * @param param_type Parameter type: PARAM_CNT_STATE, PARAM_CNT_CB, or PARAM_CNT_PHONE.
 * @param n_skip Number of utterances to skip (0 for none).
 * @param run_len Number of utterances to process (-1 for all).
 * @param part Corpus part number (0 for none).
 * @param n_part Total number of corpus parts (0 for none).
 * @return 0 on success, -1 on error.
 */
int pstrain_param_cnt(const char *mdef_path,
                  const char *dict_path,
                  const char *fdict_path,
                  const char *ctl_path,
                  const char *lsn_path,
                  const char *ts2cb_path,
                  const char *seg_dir,
                  const char *seg_ext,
                  const char *output_path,
                  int32 param_type,
                  uint32 n_skip,
                  int32 run_len,
                  uint32 part,
                  uint32 n_part);

#endif /* PSTRAIN_PARAM_CNT_H */
