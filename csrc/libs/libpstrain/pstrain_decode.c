#include "pstrain_decode.h"

#include <pocketsphinx.h>
#include <stdlib.h>

struct pstrain_decoder_s {
    ps_decoder_t *ps;
};

static int
set_str(ps_config_t *config, const char *name, const char *value)
{
    return value == NULL || ps_config_set_str(config, name, value) != NULL;
}

static int
set_float(ps_config_t *config, const char *name, double value)
{
    return ps_config_set_float(config, name, value) != NULL;
}

pstrain_decoder_t *
pstrain_decoder_create(const pstrain_decoder_config_t *options)
{
    ps_config_t *config;
    pstrain_decoder_t *decoder;

    if (options == NULL || (config = ps_config_init(NULL)) == NULL)
        return NULL;
    if (!set_str(config, "hmm", options->hmm)
        || !set_str(config, "dict", options->dict)
        || !set_str(config, "fdict", options->fdict)
        || !set_str(config, "lm", options->lm)
        || !set_float(config, "beam", options->beam)
        || !set_float(config, "wbeam", options->wbeam)
        || !set_float(config, "lw", options->lw)
        || !set_float(config, "fwdflatlw", options->lw)
        || !set_float(config, "bestpathlw", options->lw)
        || !set_float(config, "wip", options->wip)
        || !set_float(config, "pbeam", options->pbeam)
        || !set_float(config, "lpbeam", options->lpbeam)
        || !set_float(config, "lponlybeam", options->lponlybeam)
        || !set_float(config, "fwdflatbeam", options->fwdflatbeam)
        || !set_float(config, "fwdflatwbeam", options->fwdflatwbeam)
        || ps_config_set_int(config, "pl_window", options->pl_window) == NULL
        || ps_config_set_int(config, "samprate", options->samprate) == NULL
        || !set_str(config, "cmn", "batch")
        || !set_str(config, "cmninit", "40,3,-1")
        || ps_config_set_bool(config, "varnorm", 0) == NULL
        || !set_str(config, "agc", "none")) {
        ps_config_free(config);
        return NULL;
    }
    decoder = calloc(1, sizeof(*decoder));
    if (decoder != NULL)
        decoder->ps = ps_init(config);
    ps_config_free(config);
    if (decoder == NULL || decoder->ps == NULL) {
        free(decoder);
        return NULL;
    }
    return decoder;
}

int pstrain_decoder_start_utt(pstrain_decoder_t *decoder)
{ return decoder == NULL ? -1 : ps_start_utt(decoder->ps); }

int
pstrain_decoder_process_raw(pstrain_decoder_t *decoder, const int16_t *samples,
                            size_t nsamples, int no_search, int full_utt)
{
    return decoder == NULL ? -1
        : ps_process_raw(decoder->ps, samples, nsamples, no_search, full_utt);
}

int pstrain_decoder_end_utt(pstrain_decoder_t *decoder)
{ return decoder == NULL ? -1 : ps_end_utt(decoder->ps); }

const char *pstrain_decoder_hyp(pstrain_decoder_t *decoder)
{ return decoder == NULL ? NULL : ps_get_hyp(decoder->ps, NULL); }

void
pstrain_decoder_free(pstrain_decoder_t *decoder)
{
    if (decoder != NULL) {
        ps_free(decoder->ps);
        free(decoder);
    }
}

const char *pstrain_pocketsphinx_version(void)
{ return PSTRAIN_POCKETSPHINX_VERSION "+" PSTRAIN_POCKETSPHINX_COMMIT; }
