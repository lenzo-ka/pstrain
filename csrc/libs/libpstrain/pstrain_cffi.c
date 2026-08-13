/* Stable, project-namespaced CFFI entry points for vendored SphinxBase code. */

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/fe.h>
#include <sphinxbase/logmath.h>

void pstrain_cffi_ckd_free(void *ptr) { ckd_free(ptr); }
void pstrain_cffi_ckd_free_3d(void *ptr) { ckd_free_3d(ptr); }

void pstrain_cffi_fe_start_stream(fe_t *fe) { fe_start_stream(fe); }
int pstrain_cffi_fe_start_utt(fe_t *fe) { return fe_start_utt(fe); }
int pstrain_cffi_fe_process_frames(fe_t *fe, int16 const **spch, size_t *nsamps,
                                   mfcc_t **cep, int32 *nframes, int32 *frameidx)
{
    return fe_process_frames(fe, spch, nsamps, cep, nframes, frameidx);
}
int pstrain_cffi_fe_end_utt(fe_t *fe, mfcc_t *cep, int32 *nframes)
{
    return fe_end_utt(fe, cep, nframes);
}
int pstrain_cffi_fe_free(fe_t *fe) { return fe_free(fe); }
int pstrain_cffi_fe_get_output_size(fe_t *fe) { return fe_get_output_size(fe); }
int pstrain_cffi_fe_mfcc_to_float(fe_t *fe, mfcc_t **input, float32 **output,
                                  int32 nframes)
{
    return fe_mfcc_to_float(fe, input, output, nframes);
}

logmath_t *pstrain_cffi_logmath_init(float64 base, int shift, int use_table)
{
    return logmath_init(base, shift, use_table);
}
void pstrain_cffi_logmath_free(logmath_t *lmath) { logmath_free(lmath); }
int32 pstrain_cffi_logmath_log(logmath_t *lmath, float64 p)
{
    return logmath_log(lmath, p);
}
float64 pstrain_cffi_logmath_exp(logmath_t *lmath, int32 p)
{
    return logmath_exp(lmath, p);
}
int32 pstrain_cffi_logmath_add(logmath_t *lmath, int32 p, int32 q)
{
    return logmath_add(lmath, p, q);
}
float64 pstrain_cffi_logmath_get_base(logmath_t *lmath)
{
    return logmath_get_base(lmath);
}
