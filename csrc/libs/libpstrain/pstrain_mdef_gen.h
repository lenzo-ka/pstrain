/**
 * @file pstrain_mdef_gen.h
 * @brief Simplified API for generating mdef files.
 *
 * Wraps mk_mdef_gen functionality for CFFI binding.
 */

#ifndef PSTRAIN_MDEF_GEN_H
#define PSTRAIN_MDEF_GEN_H

#include <sphinxbase/prim_type.h>

/**
 * Generate CI (context-independent) mdef from phone list.
 *
 * @param phone_list_path Path to phone list file (one phone per line)
 * @param output_path Output mdef file path
 * @param n_state Number of emitting states per phone (typically 3)
 * @return 0 on success, -1 on error
 */
int pstrain_mdef_gen_ci(const char *phone_list_path,
                    const char *output_path,
                    uint32 n_state);

/**
 * Generate all-triphones mdef from dictionary.
 *
 * Creates mdef with all possible triphones from dictionary entries.
 *
 * @param phone_list_path Path to CI phone list
 * @param dict_path Path to pronunciation dictionary
 * @param filler_dict_path Path to filler dictionary (optional, NULL if none)
 * @param output_path Output mdef file path
 * @param n_state Number of emitting states per phone
 * @param ignore_wpos If true, ignore word position in triphones
 * @return 0 on success, -1 on error
 */
int pstrain_mdef_gen_alltriphones(const char *phone_list_path,
                              const char *dict_path,
                              const char *filler_dict_path,
                              const char *output_path,
                              uint32 n_state,
                              int32 ignore_wpos);

/**
 * Generate untied mdef from transcripts.
 *
 * Creates mdef with triphones observed in transcripts, pruned by
 * occurrence threshold.
 *
 * @param phone_list_path Path to CI phone list
 * @param dict_path Path to pronunciation dictionary
 * @param filler_dict_path Path to filler dictionary (optional)
 * @param transcript_path Path to transcript file
 * @param output_path Output mdef file path
 * @param n_state Number of emitting states per phone
 * @param ignore_wpos If true, ignore word position
 * @return 0 on success, -1 on error
 */
int pstrain_mdef_gen_untied(const char *phone_list_path,
                        const char *dict_path,
                        const char *filler_dict_path,
                        const char *transcript_path,
                        const char *output_path,
                        uint32 n_state,
                        int32 ignore_wpos);

/**
 * Count triphones in transcripts.
 *
 * @param phone_list_path Path to CI phone list
 * @param dict_path Path to pronunciation dictionary
 * @param filler_dict_path Path to filler dictionary (optional)
 * @param transcript_path Path to transcript file
 * @param output_path Output counts file path
 * @param ignore_wpos If true, ignore word position
 * @return 0 on success, -1 on error
 */
int pstrain_mdef_count_triphones(const char *phone_list_path,
                             const char *dict_path,
                             const char *filler_dict_path,
                             const char *transcript_path,
                             const char *output_path,
                             int32 ignore_wpos);

#endif /* PSTRAIN_MDEF_GEN_H */
