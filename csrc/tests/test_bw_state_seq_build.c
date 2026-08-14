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

static int
exercise_boundary_mode(pstrain_bw_context_t *ctx, int multipron,
                       uint32 *n_state_out, uint32 *first_phone, uint32 *last_phone)
{
    state_t *states;
    uint32 n_state = 0;

    CHECK(pstrain_bw_set_multipron(ctx, multipron) == 0, "select boundary builder mode");
    states = pstrain_bw_build_state_seq(ctx, "<s> a and </s>", &n_state);
    CHECK(states != NULL, "boundary sequence builds");
    CHECK(n_state > 1, "boundary sequence has states");
    *n_state_out = n_state;
    *first_phone = states[0].phn;
    *last_phone = states[n_state - 2].phn;
    pstrain_bw_free_state_seq(states, n_state);
    return 0;
}

int
main(int argc, char *argv[])
{
    pstrain_bw_config_t config;
    pstrain_bw_context_t *ctx;
    uint32 off_n[2], on_n[2], off_first[2], on_first[2], off_last[2], on_last[2];
    int multipron;

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
    for (multipron = 0; multipron <= 1; ++multipron) {
        CHECK(exercise_boundary_mode(ctx, multipron, &off_n[multipron],
                                     &off_first[multipron], &off_last[multipron]) == 0,
              "off mode retains boundaries");
    }
    pstrain_bw_free(ctx);

    config.optional_boundary_silence = 1;
    ctx = pstrain_bw_init(argv[1], argv[2], argv[3], argv[4], argv[5], &config);
    CHECK(ctx != NULL, "initialize optional-boundary BW fixture");
    CHECK(pstrain_bw_set_dict(ctx, argv[6], argv[7]) == 0, "load optional-boundary dictionaries");
    for (multipron = 0; multipron <= 1; ++multipron) {
        CHECK(exercise_boundary_mode(ctx, multipron, &on_n[multipron],
                                     &on_first[multipron], &on_last[multipron]) == 0,
              "on mode bypasses boundaries");
        CHECK(on_n[multipron] < off_n[multipron], "on mode removes boundary HMM states");
        CHECK(on_first[multipron] != off_first[multipron], "initial SIL is bypassed");
        CHECK(on_last[multipron] != off_last[multipron], "final SIL is bypassed");
    }
    pstrain_bw_free(ctx);
    printf("PASS: public BW state-sequence build modes survive injected failure\n");
    return 0;
}
