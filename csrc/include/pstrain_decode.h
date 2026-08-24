#ifndef PSTRAIN_DECODE_H
#define PSTRAIN_DECODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pstrain_decoder_s pstrain_decoder_t;

typedef struct {
    const char *hmm;
    const char *dict;
    const char *fdict;
    const char *lm;
    double beam;
    double wbeam;
    double lw;
    double wip;
    double pbeam;
    double lpbeam;
    double lponlybeam;
    double fwdflatbeam;
    double fwdflatwbeam;
    long pl_window;
    long samprate;
    const char *agc;
    const char *cmn;
    const char *cmninit;
    int varnorm;
    int remove_noise;
    long topn; /* top-N Gaussians per state; 0 = leave PocketSphinx default */
} pstrain_decoder_config_t;

pstrain_decoder_t *pstrain_decoder_create(const pstrain_decoder_config_t *config);
int pstrain_decoder_start_utt(pstrain_decoder_t *decoder);
int pstrain_decoder_process_raw(pstrain_decoder_t *decoder,
                                const int16_t *samples, size_t nsamples,
                                int no_search, int full_utt);
int pstrain_decoder_end_utt(pstrain_decoder_t *decoder);
const char *pstrain_decoder_hyp(pstrain_decoder_t *decoder);
const char *pstrain_decoder_config_str(pstrain_decoder_t *decoder, const char *name);
long pstrain_decoder_config_int(pstrain_decoder_t *decoder, const char *name);
void pstrain_decoder_free(pstrain_decoder_t *decoder);
const char *pstrain_pocketsphinx_version(void);

#ifdef __cplusplus
}
#endif

#endif
