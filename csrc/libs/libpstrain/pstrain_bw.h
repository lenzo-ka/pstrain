/**
 * @file pstrain_bw.h
 * @brief Simplified Baum-Welch training API for CFFI
 */

#ifndef PSTRAIN_BW_H
#define PSTRAIN_BW_H

#include <sphinxbase/prim_type.h>

#define PSTRAIN_BW_FINAL_STATE_NOT_REACHED -2

typedef enum pstrain_bw_unobserved_gaussian_policy_e {
    PSTRAIN_BW_UNOBSERVED_GAUSSIAN_INVALID = 0,
    PSTRAIN_BW_UNOBSERVED_GAUSSIAN_ZERO = 1,
    PSTRAIN_BW_UNOBSERVED_GAUSSIAN_RETAIN = 2
} pstrain_bw_unobserved_gaussian_policy_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Training configuration
 */
typedef struct pstrain_bw_config_s {
    float64 a_beam;      /**< Forward beam (default: 1e-90) */
    float64 b_beam;      /**< Backward beam (default: 1e-10) */
    uint32 topn;         /**< Gaussians evaluated per codebook (default: 1) */
    float32 spthresh;    /**< State pruning threshold (default: 0) */
    float32 mixw_floor;  /**< Mixture-weight floor (default: 1e-8) */
    float32 tmat_floor;  /**< Transition-probability floor (default: 1e-4) */
    int32 mixw_reest;    /**< Re-estimate mixture weights (default: 1) */
    int32 tmat_reest;    /**< Re-estimate transition matrices (default: 1) */
    int32 mean_reest;    /**< Re-estimate means (default: 1) */
    int32 var_reest;     /**< Re-estimate variances (default: 1) */
    int32 pass2var;      /**< Use 2-pass variance estimation (default: 1) */
    pstrain_bw_unobserved_gaussian_policy_t unobserved_gaussian_policy;
} pstrain_bw_config_t;

/**
 * Opaque training context
 */
typedef struct pstrain_bw_context_s pstrain_bw_context_t;
typedef struct state_s state_t;

/**
 * Initialize BW training context
 *
 * @param mdef_path Path to model definition file
 * @param means_path Path to means file
 * @param vars_path Path to variances file
 * @param mixw_path Path to mixture weights file
 * @param tmat_path Path to transition matrices file
 * @param config Training configuration; must explicitly select an
 *               unobserved-Gaussian policy
 * @return Training context, or NULL on error
 */
pstrain_bw_context_t *
pstrain_bw_init(const char *mdef_path,
            const char *means_path,
            const char *vars_path,
            const char *mixw_path,
            const char *tmat_path,
            const pstrain_bw_config_t *config);

/**
 * Free training context
 */
void
pstrain_bw_free(pstrain_bw_context_t *ctx);

/**
 * Load dictionary (lexicon) for transcript processing
 *
 * Must be called before pstrain_bw_process_utt_text.
 * Dictionary handles multiple pronunciations automatically.
 *
 * @param ctx Training context
 * @param dict_path Path to main dictionary
 * @param filler_dict_path Path to filler dictionary (may be NULL)
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_set_dict(pstrain_bw_context_t *ctx,
                const char *dict_path,
                const char *filler_dict_path);

/**
 * Enable or disable multi-pronunciation training.
 *
 * On by default. When enabled, each utterance HMM is built as a graph
 * with parallel paths for every variant of every word, and forward-
 * backward sums posteriors across variants. When disabled, the trainer
 * falls back to the historical linear path where the first listed
 * variant is always selected (matches SphinxTrain's default behavior
 * and is bit-identical to pstrain's pre-multipron output).
 *
 * @param ctx Training context
 * @param enable Nonzero to enable multipron training, zero to disable
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_set_multipron(pstrain_bw_context_t *ctx, int enable);

/** Set the forward pruning beam for a retry, returning the previous value. */
float64
pstrain_bw_set_a_beam(pstrain_bw_context_t *ctx, float64 a_beam);

/** Build an utterance state sequence for structural inspection.
 * The returned sequence is caller-owned in both graph and linear modes and
 * must be released exactly once with pstrain_bw_free_state_seq().
 */
state_t *
pstrain_bw_build_state_seq(pstrain_bw_context_t *ctx,
                           const char *transcript,
                           uint32 *n_state);

void
pstrain_bw_free_state_seq(state_t *state_seq, uint32 n_state);

/** Internal ownership adapter, exposed for the C regression harness. */
state_t *pstrain_bw_copy_state_seq(const state_t *source, uint32 n_state);

/** Number of CI senones used as primary fallback models in this pass. */
uint32 pstrain_bw_count_active_fallback_senones(pstrain_bw_context_t *ctx);

/** Whether a senone was activated as a primary CI fallback this pass. */
int pstrain_bw_fallback_senone_active(pstrain_bw_context_t *ctx, uint32 senone);

/**
 * Process utterance with transcript text
 *
 * Dictionary must be loaded first via pstrain_bw_set_dict.
 * Handles word lookup, multiple pronunciations, and state sequence.
 *
 * @param ctx Training context
 * @param features Feature vectors (n_frames * feat_dim, row-major)
 * @param n_frames Number of frames
 * @param transcript Word transcript (space-separated, uppercase)
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_process_utt_text(pstrain_bw_context_t *ctx,
                        const float *features,
                        uint32 n_frames,
                        const char *transcript);

/**
 * Process utterance from raw MFCC features (13-dim)
 *
 * Uses C feat module to apply CMN and compute deltas, exactly like SphinxTrain.
 * Dictionary must be loaded first via pstrain_bw_set_dict.
 *
 * @param ctx Training context
 * @param mfcc Raw MFCC features (n_frames * 13, row-major)
 * @param n_mfcc_frames Number of MFCC frames
 * @param transcript Word transcript (space-separated, uppercase)
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_process_utt_mfcc(pstrain_bw_context_t *ctx,
                        const float *mfcc,
                        uint32 n_mfcc_frames,
                        const char *transcript);

/**
 * Process a single utterance (low-level, with phone IDs)
 *
 * @param ctx Training context
 * @param features Feature vectors (n_frames * feat_dim, row-major)
 * @param n_frames Number of frames
 * @param phone_ids Phone ID sequence for this utterance
 * @param n_phones Number of phones
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_process_utt(pstrain_bw_context_t *ctx,
                   const float *features,
                   uint32 n_frames,
                   const uint32 *phone_ids,
                   uint32 n_phones);

/**
 * Normalize accumulators and update model
 *
 * Call after processing all utterances in an iteration.
 *
 * @param ctx Training context
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_normalize(pstrain_bw_context_t *ctx);

/** Serialize native BW accumulators in the upstream accum_dump format. */
int pstrain_bw_dump_accum(pstrain_bw_context_t *ctx, const char *accum_dir);

/** Merge upstream accumulator directories with the vendored norm rdacc path. */
int pstrain_bw_restore_accumdirs(pstrain_bw_context_t *ctx,
                                 const char *const *accum_dirs,
                                 uint32 n_accum_dirs);

/**
 * Save trained model
 *
 * @param ctx Training context
 * @param means_path Output path for means
 * @param vars_path Output path for variances
 * @param mixw_path Output path for mixture weights
 * @param tmat_path Output path for transition matrices
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_save(pstrain_bw_context_t *ctx,
            const char *means_path,
            const char *vars_path,
            const char *mixw_path,
            const char *tmat_path);

/**
 * Get training statistics
 *
 * @param ctx Training context
 * @param total_log_lik Output: total log likelihood (may be NULL)
 * @param total_frames Output: total frames processed (may be NULL)
 * @param total_utts Output: total utterances processed (may be NULL)
 */
void
pstrain_bw_get_stats(pstrain_bw_context_t *ctx,
                 float64 *total_log_lik,
                 uint32 *total_frames,
                 uint32 *total_utts);

/**
 * Save density counts (for Gaussian splitting)
 *
 * @param ctx Training context
 * @param counts_path Output path for density counts
 * @return 0 on success, -1 on error
 */
int
pstrain_bw_save_counts(pstrain_bw_context_t *ctx, const char *counts_path);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_BW_H */
