/**
 * @file pstrain_align.h
 * @brief In-process forced alignment API for pstrain (CFFI binding target).
 *
 * Thin session wrapper over the sphinx3 forced aligner vendored under
 * csrc/programs/sphinx3_align. Replaces the subprocess-based wrapper in
 * pstrain/lib/alignment/sphinx3.py and the PocketSphinx-based wrapper in
 * pstrain/lib/alignment/core.py.
 *
 * Lifetime: one aligner instance per process. The underlying C aligner
 * holds module-static state; a second concurrent pstrain_align_init while a
 * context is still alive returns NULL.
 */

#ifndef PSTRAIN_ALIGN_H
#define PSTRAIN_ALIGN_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pstrain_align_config_s {
    double  beam;            /**< Main pruning beam.        Default 1e-64. */
    int     insert_sil;      /**< Insert optional silences. Default 1.    */
    int     compute_phones;  /**< Return phone segments.    Default 1.    */
    int     compute_states;  /**< Return state segments.    Default 0.    */
    const char *feat_type;   /**< Feature stream spec.   Default "1s_c_d_dd". */
    const char *cmn;         /**< CMN type.              Default "current". */
    const char *cmninit;     /**< Initial CMN vector.    Default "40,3,-1". */
    const char *agc;         /**< AGC type.              Default "none".    */
    int     varnorm;         /**< Cepstral variance norm.   Default 0.    */
    int     ceplen;          /**< Cepstral vector width.    Default 13.   */
    int     frate;           /**< Frame rate (Hz).          Default 100.  */
    int     lts_mismatch;    /**< Use LTS rules for OOV.    Default 0.    */
    int     verbatim_tokens; /**< Honor explicit WORD(n).    Default 0.    */
} pstrain_align_config_t;

#define PSTRAIN_ABI_VERSION 4

uint32 pstrain_abi_version(void);
void pstrain_align_config_default(pstrain_align_config_t *config);

typedef struct pstrain_align_context_s pstrain_align_context_t;

typedef struct pstrain_align_seg_s {
    const char *name;    /**< Word/phone/state label (owned by result). */
    int32 start_frame;
    int32 end_frame;
    int32 score;
} pstrain_align_seg_t;

typedef struct pstrain_align_result_s {
    pstrain_align_seg_t *words;
    uint32 n_words;
    pstrain_align_seg_t *phones;
    uint32 n_phones;
    pstrain_align_seg_t *states;
    uint32 n_states;
    int32 total_score;
    int32 n_frames;
    void   *_arena;        /**< Internal: string storage. Don't touch. */
} pstrain_align_result_t;

/**
 * Initialize a forced-alignment session.
 *
 * @param mdef_path Model definition file.
 * @param mean_path Means file.
 * @param var_path  Variances file.
 * @param mixw_path Mixture weights file.
 * @param tmat_path Transition matrices file.
 * @param feat_params_path Path retained for CLI parity diagnostics. The
 *        caller must project validated feature parameters into config.
 * @param dict_path Main dictionary.
 * @param fdict_path Filler dictionary (may be NULL).
 * @param config Tunables (NULL for defaults).
 * @return Opaque context, or NULL on failure (see pstrain_align_last_error).
 */
pstrain_align_context_t *
pstrain_align_init(const char *mdef_path,
               const char *mean_path,
               const char *var_path,
               const char *mixw_path,
               const char *tmat_path,
               const char *feat_params_path,
               const char *dict_path,
               const char *fdict_path,
               const pstrain_align_config_t *config);

/**
 * Tear down a forced-alignment session.
 */
void pstrain_align_free(pstrain_align_context_t *ctx);

/**
 * Set the live pruning beam and return its previous value.
 */
double pstrain_align_set_beam(pstrain_align_context_t *ctx, double beam);

/**
 * Align one utterance from already-extracted MFCC frames.
 *
 * @param ctx Context.
 * @param mfcc Row-major MFCC matrix, shape (n_frames, ncep).
 * @param n_frames Number of MFCC frames.
 * @param ncep Number of cepstral coefficients per frame.
 * @param transcript Reference transcript (word sequence, may include the
 *        usual sphinx <s>/</s> markers; they will be stripped).
 * @param utt_id Utterance id (for logging; may be NULL).
 * @param out_result Out: result struct. Free with pstrain_align_result_free.
 * @return 0 on success, negative on failure.
 */
int
pstrain_align_mfcc(pstrain_align_context_t *ctx,
               const float *mfcc,
               uint32 n_frames,
               uint32 ncep,
               const char *transcript,
               const char *utt_id,
               pstrain_align_result_t **out_result);

/**
 * Align one utterance from a cepstrum file on disk (.mfc / sphinx2 binary
 * cepstra). Convenient for parity-checking against the standalone
 * sphinx3_align binary.
 */
int
pstrain_align_mfc_file(pstrain_align_context_t *ctx,
                   const char *mfc_path,
                   const char *transcript,
                   const char *utt_id,
                   pstrain_align_result_t **out_result);

/**
 * Free a result struct returned by pstrain_align_mfcc / pstrain_align_mfc_file.
 */
void pstrain_align_result_free(pstrain_align_result_t *result);

/**
 * Return the most recent error message recorded by pstrain_align, or NULL if
 * nothing has gone wrong. Pointer is owned by the library; valid until
 * the next pstrain_align_* call.
 */
const char *pstrain_align_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_ALIGN_H */
