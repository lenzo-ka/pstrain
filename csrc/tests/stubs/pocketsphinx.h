#ifndef TEST_STUB_POCKETSPHINX_H
#define TEST_STUB_POCKETSPHINX_H

#include <stddef.h>
#include <stdint.h>

typedef struct ps_config_s ps_config_t;
typedef struct ps_decoder_s ps_decoder_t;

ps_config_t *ps_config_init(const void *defaults);
void ps_config_free(ps_config_t *config);
ps_config_t *ps_config_set_str(ps_config_t *config, const char *name,
                               const char *value);
ps_config_t *ps_config_set_float(ps_config_t *config, const char *name,
                                 double value);
ps_config_t *ps_config_set_int(ps_config_t *config, const char *name,
                               long value);
ps_config_t *ps_config_set_bool(ps_config_t *config, const char *name,
                                int value);
const char *ps_config_str(ps_config_t *config, const char *name);
long ps_config_int(ps_config_t *config, const char *name);
int ps_config_bool(ps_config_t *config, const char *name);

ps_decoder_t *ps_init(ps_config_t *config);
int ps_free(ps_decoder_t *decoder);
ps_config_t *ps_get_config(ps_decoder_t *decoder);
int ps_start_stream(ps_decoder_t *decoder);
int ps_start_utt(ps_decoder_t *decoder);
int ps_process_raw(ps_decoder_t *decoder, const int16_t *samples,
                   size_t nsamples, int no_search, int full_utt);
int ps_end_utt(ps_decoder_t *decoder);
const char *ps_get_hyp(ps_decoder_t *decoder, int32_t *out_best_score);

#endif
