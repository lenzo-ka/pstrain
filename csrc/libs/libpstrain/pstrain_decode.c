#include "pstrain_decode.h"

#include <pocketsphinx.h>
#include <stdlib.h>

struct pstrain_decoder_s {
    ps_decoder_t *ps;
    uint64_t native_init_generation;
};

static uint64_t native_init_generation;

static int
init_native_decoder(pstrain_decoder_t *decoder, ps_config_t *config)
{
    decoder->ps = ps_init(config);
    if (decoder->ps == NULL)
        return 0;
    decoder->native_init_generation = ++native_init_generation;
    return 1;
}

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
        || !set_str(config, "cmn", options->cmn)
        || !set_str(config, "cmninit", options->cmninit)
        || ps_config_set_bool(config, "varnorm", options->varnorm) == NULL
        || ps_config_set_bool(config, "remove_noise", options->remove_noise) == NULL
        || !set_str(config, "agc", options->agc)) {
        ps_config_free(config);
        return NULL;
    }
    decoder = calloc(1, sizeof(*decoder));
    if (decoder != NULL)
        (void)init_native_decoder(decoder, config);
    ps_config_free(config);
    if (decoder == NULL || decoder->ps == NULL) {
        free(decoder);
        return NULL;
    }
    return decoder;
}

int
pstrain_decoder_start_utt(pstrain_decoder_t *decoder)
{
    if (decoder == NULL)
        return -1;
    if (ps_config_bool(ps_get_config(decoder->ps), "remove_noise")
        && ps_start_stream(decoder->ps) < 0)
        return -1;
    return ps_start_utt(decoder->ps);
}

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

const char *
pstrain_decoder_config_str(pstrain_decoder_t *decoder, const char *name)
{
    return decoder == NULL ? NULL : ps_config_str(ps_get_config(decoder->ps), name);
}

long
pstrain_decoder_config_int(pstrain_decoder_t *decoder, const char *name)
{
    return decoder == NULL ? 0 : ps_config_int(ps_get_config(decoder->ps), name);
}

uint64_t
pstrain_decoder_native_init_generation(pstrain_decoder_t *decoder)
{
    return decoder == NULL ? 0 : decoder->native_init_generation;
}

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
