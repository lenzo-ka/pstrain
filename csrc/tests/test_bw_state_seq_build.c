/* Exercise the public state-sequence builder and its failure cleanup. */

#include <stdio.h>
#include <math.h>
#include <string.h>

#include <s3/state.h>

#include "../libs/libpstrain/pstrain_bw.h"
#include "../programs/bw/next_utt_states.h"

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
                       int optional_final_silence, uint32 *n_state_out,
                       uint32 *n_initial, uint32 *final_priors)
{
    state_t *states;
    uint32 n_state = 0;

    CHECK(pstrain_bw_set_multipron(ctx, multipron) == 0, "select boundary builder mode");
    states = pstrain_bw_build_state_seq(ctx, "<s> a and </s>", &n_state);
    CHECK(states != NULL, "boundary sequence builds");
    CHECK(n_state > 1, "boundary sequence has states");
    *n_state_out = n_state;
    *n_initial = 1;
    CHECK(states[0].mixw != TYING_NON_EMITTING,
          "state zero is the sole frame-zero emitting entry");
    for (uint32 i = 1; i < n_state; ++i) {
        uint32 p;
        if (states[i].mixw == TYING_NON_EMITTING)
            continue;
        for (p = 0; p < states[i].n_prior; ++p)
            if (states[i].prior_state[p] != i)
                break;
        if (p == states[i].n_prior)
            ++*n_initial;
    }
    CHECK(*n_initial == 1,
          "graph has exactly one frame-zero emitting entry");
    *final_priors = states[n_state - 1].n_prior;
    if (optional_final_silence) {
        uint32 i, j, bypass_predecessors = 0;
        for (i = 0; i < states[n_state - 1].n_prior; ++i) {
            uint32 pred = states[n_state - 1].prior_state[i];
            float32 bypass_mass = 0.0f, retained_mass = 0.0f;
            if (states[pred].mixw != TYING_NON_EMITTING || states[pred].n_next < 2)
                continue;
            for (j = 0; j < states[pred].n_next; ++j) {
                if (states[pred].next_state[j] == n_state - 1)
                    bypass_mass += states[pred].next_tprob[j];
                else
                    retained_mass += states[pred].next_tprob[j];
            }
            CHECK(fabs(bypass_mass + retained_mass - 1.0f) < 1e-6,
                  "retained and bypass final alternatives total one");
            ++bypass_predecessors;
        }
        CHECK(bypass_predecessors > 0,
              "optional final silence exposes a bypass predecessor");
    }
    pstrain_bw_free_state_seq(states, n_state);
    return 0;
}

int
main(int argc, char *argv[])
{
    pstrain_bw_config_t config;
    pstrain_bw_context_t *ctx;
    uint32 off_n[2], on_n[2], off_initial[2], on_initial[2], off_final[2], on_final[2];
    int multipron;

    CHECK(argc == 8, "expected model and dictionary fixture paths");
    CHECK(!next_utt_states_graph_built(0, 0),
          "linear builder storage is static");
    CHECK(next_utt_states_graph_built(1, 0),
          "multipron graph storage is owned");
    CHECK(next_utt_states_graph_built(0, 1),
          "optional-final graph storage is owned");
    CHECK(next_utt_states_graph_built(1, 1),
          "combined graph storage is owned");
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
        CHECK(exercise_boundary_mode(ctx, multipron, 0, &off_n[multipron],
                                     &off_initial[multipron], &off_final[multipron]) == 0,
              "off mode retains boundaries");
    }
    pstrain_bw_free(ctx);

    config.optional_final_silence = 1;
    ctx = pstrain_bw_init(argv[1], argv[2], argv[3], argv[4], argv[5], &config);
    CHECK(ctx != NULL, "initialize optional-final BW fixture");
    CHECK(pstrain_bw_set_dict(ctx, argv[6], argv[7]) == 0, "load optional-final dictionaries");
    for (multipron = 0; multipron <= 1; ++multipron) {
        CHECK(exercise_mode(ctx, multipron) == 0, "optional-final failure cleanup");
        CHECK(exercise_boundary_mode(ctx, multipron, 1, &on_n[multipron],
                                     &on_initial[multipron], &on_final[multipron]) == 0,
              "on mode bypasses final silence");
        CHECK(on_n[multipron] == off_n[multipron], "on mode retains boundary HMM states");
        CHECK(off_initial[multipron] == 1, "off mode has exactly one initial state");
        CHECK(on_initial[multipron] == 1, "optional final silence keeps one initial state");
        CHECK(on_final[multipron] > off_final[multipron], "final SIL has bypass predecessors");
    }
    pstrain_bw_free(ctx);
    printf("PASS: public BW state-sequence build modes survive injected failure\n");
    return 0;
}
