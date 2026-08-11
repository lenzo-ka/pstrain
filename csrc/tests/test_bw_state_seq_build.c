/* Exercise the public state-sequence builder and its failure cleanup. */

#include <stdio.h>
#include <string.h>

#include <s3/state.h>

#include "../libs/libpstrain/pstrain_bw.h"

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int
exercise_mode(pstrain_bw_context_t *ctx, int multipron)
{
    state_t *states;
    uint32 n_state = 0;

    CHECK(pstrain_bw_set_multipron(ctx, multipron) == 0, "select builder mode");

    /* UNKNOWN occurs after a valid word, exercising partial construction and
     * cleanup rather than argument validation or a first-token rejection. */
    states = pstrain_bw_build_state_seq(ctx, "<s> a UNKNOWN </s>", &n_state);
    CHECK(states == NULL, "mid-utterance lookup failure is reported");

    n_state = 0;
    states = pstrain_bw_build_state_seq(ctx, "<s> a and </s>", &n_state);
    CHECK(states != NULL, "valid sequence builds after failure");
    CHECK(n_state > 0, "valid sequence has states");
    pstrain_bw_free_state_seq(states, n_state);
    return 0;
}

int
main(int argc, char *argv[])
{
    pstrain_bw_config_t config;
    pstrain_bw_context_t *ctx;

    CHECK(argc == 8, "expected model and dictionary fixture paths");
    memset(&config, 0, sizeof(config));
    config.a_beam = 1e-90;
    config.b_beam = 1e-10;
    config.topn = 1;
    config.mixw_floor = 1e-8f;
    config.tmat_floor = 1e-4f;
    config.pass2var = 1;
    config.unobserved_gaussian_policy = PSTRAIN_BW_UNOBSERVED_GAUSSIAN_ZERO;

    ctx = pstrain_bw_init(argv[1], argv[2], argv[3], argv[4], argv[5], &config);
    CHECK(ctx != NULL, "initialize BW fixture");
    CHECK(pstrain_bw_set_dict(ctx, argv[6], argv[7]) == 0, "load dictionaries");
    CHECK(exercise_mode(ctx, 0) == 0, "linear mode");
    CHECK(exercise_mode(ctx, 1) == 0, "graph mode");
    pstrain_bw_free(ctx);
    printf("PASS: public BW state-sequence build modes survive injected failure\n");
    return 0;
}
