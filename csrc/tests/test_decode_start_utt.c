/* Contract coverage for conditional stream reset and failure propagation. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pstrain_decode.h"
#include "pocketsphinx.h"

struct ps_config_s {
    int remove_noise;
};

struct ps_decoder_s {
    ps_config_t config;
};

static int start_stream_result;
static int start_stream_calls;
static int start_utt_calls;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

ps_config_t *
ps_config_init(const void *defaults)
{
    ps_config_t *config;
    (void)defaults;
    config = calloc(1, sizeof(*config));
    return config;
}

void ps_config_free(ps_config_t *config) { free(config); }

ps_config_t *
ps_config_set_str(ps_config_t *config, const char *name, const char *value)
{
    (void)name;
    (void)value;
    return config;
}

ps_config_t *
ps_config_set_float(ps_config_t *config, const char *name, double value)
{
    (void)name;
    (void)value;
    return config;
}

ps_config_t *
ps_config_set_int(ps_config_t *config, const char *name, long value)
{
    (void)name;
    (void)value;
    return config;
}

ps_config_t *
ps_config_set_bool(ps_config_t *config, const char *name, int value)
{
    if (strcmp(name, "remove_noise") == 0)
        config->remove_noise = value;
    return config;
}

const char *ps_config_str(ps_config_t *config, const char *name)
{
    (void)config;
    (void)name;
    return "";
}

long ps_config_int(ps_config_t *config, const char *name)
{
    (void)config;
    (void)name;
    return 0;
}

int ps_config_bool(ps_config_t *config, const char *name)
{
    CHECK(config != NULL, "config is available");
    CHECK(strcmp(name, "remove_noise") == 0, "remove_noise is queried");
    return config->remove_noise;
}

ps_decoder_t *
ps_init(ps_config_t *config)
{
    ps_decoder_t *decoder = calloc(1, sizeof(*decoder));
    if (decoder != NULL)
        decoder->config = *config;
    return decoder;
}

int ps_free(ps_decoder_t *decoder)
{
    free(decoder);
    return 0;
}

ps_config_t *ps_get_config(ps_decoder_t *decoder) { return &decoder->config; }

int ps_start_stream(ps_decoder_t *decoder)
{
    (void)decoder;
    ++start_stream_calls;
    return start_stream_result;
}

int ps_start_utt(ps_decoder_t *decoder)
{
    (void)decoder;
    ++start_utt_calls;
    return 0;
}

int ps_process_raw(ps_decoder_t *decoder, const int16_t *samples,
                   size_t nsamples, int no_search, int full_utt)
{
    (void)decoder;
    (void)samples;
    (void)nsamples;
    (void)no_search;
    (void)full_utt;
    return 0;
}

int ps_end_utt(ps_decoder_t *decoder)
{
    (void)decoder;
    return 0;
}

const char *ps_get_hyp(ps_decoder_t *decoder, int32_t *out_best_score)
{
    (void)decoder;
    (void)out_best_score;
    return "hypothesis";
}

static pstrain_decoder_t *
new_decoder(int remove_noise)
{
    pstrain_decoder_config_t options = {0};
    options.remove_noise = remove_noise;
    return pstrain_decoder_create(&options);
}

int
main(void)
{
    pstrain_decoder_t *decoder;

    start_stream_result = -1;
    decoder = new_decoder(1);
    CHECK(decoder != NULL, "noise-enabled decoder creation");
    CHECK(pstrain_decoder_start_utt(decoder) == -1,
          "stream-start failure remains fatal when noise removal is enabled");
    CHECK(start_stream_calls == 1, "stream reset attempted");
    CHECK(start_utt_calls == 0, "utterance does not start after reset failure");
    pstrain_decoder_free(decoder);

    start_stream_calls = 0;
    start_utt_calls = 0;
    decoder = new_decoder(0);
    CHECK(decoder != NULL, "noise-disabled decoder creation");
    CHECK(pstrain_decoder_start_utt(decoder) == 0,
          "utterance starts without unavailable noise statistics");
    CHECK(start_stream_calls == 0, "stream reset skipped");
    CHECK(start_utt_calls == 1, "utterance start attempted");
    pstrain_decoder_free(decoder);

    printf("PASS: decoder stream-reset contract\n");
    return 0;
}
