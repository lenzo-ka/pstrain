/**
 * @file main_align.h
 * @brief In-process entry points exposed by sphinx3_align.
 *
 * Lets the libpstrain wrapper in csrc/libs/libpstrain/pstrain_align.c drive the aligner
 * without going through main(). The standalone CLI in main_align.c still
 * uses the same symbols; the library wrapper just bypasses the argv-parsing
 * and control/transcript-file handling.
 */

#ifndef _PSTRAIN_MAIN_ALIGN_H_
#define _PSTRAIN_MAIN_ALIGN_H_

#include <sphinxbase/prim_type.h>
#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/fe.h>

#include "kbcore.h"
#include "ascr.h"
#include "fast_algo_struct.h"
#include "adaptor.h"
#include "dict.h"
#include "s3_align.h"

#ifdef __cplusplus
extern "C" {
#endif

extern kbcore_t *kbc;
extern fe_t *fe;
extern ascr_t *ascr;
extern fast_gmm_t *fastgmm;
extern adapt_am_t *adapt_am;
extern dict_t *dict;
extern float32 ***feat;

void models_init(cmd_ln_t *config);
void models_free(void);

/**
 * Run forced alignment on a single utterance using the already-loaded
 * model and the feature frames in the module-level `feat` buffer. Returns
 * the seg lists owned by the aligner; call `align_utt_release` after
 * copying them out.
 *
 * @param sent     Reference transcript (mutable buffer).
 * @param nfr      Number of feature frames in `feat`.
 * @param uttid    Utterance id for logging.
 * @param out_stseg Out: state-level segmentation list (may be NULL).
 * @param out_phseg Out: phone-level segmentation list (may be NULL).
 * @param out_wdseg Out: word-level segmentation list (may be NULL).
 * @return 0 on success, negative on failure.
 */
int align_utt_capture(char *sent,
                      int32 nfr,
                      char *uttid,
                      align_stseg_t **out_stseg,
                      align_phseg_t **out_phseg,
                      align_wdseg_t **out_wdseg);

void align_utt_release(void);

#ifdef __cplusplus
}
#endif

#endif /* _PSTRAIN_MAIN_ALIGN_H_ */
