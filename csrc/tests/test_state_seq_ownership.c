/* Regression for the public BW state-sequence ownership contract. */

#include <stdio.h>

#include <s3/state.h>
#include <sphinxbase/ckd_alloc.h>

#include "../libs/libpstrain/pstrain_bw.h"

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int
main(void)
{
    state_t source[2] = {{0}};
    uint32 *next_state = ckd_calloc(1, sizeof(*next_state));
    float32 *next_tprob = ckd_calloc(1, sizeof(*next_tprob));
    uint32 *prior_state = ckd_calloc(1, sizeof(*prior_state));
    float32 *prior_tprob = ckd_calloc(1, sizeof(*prior_tprob));
    state_t *owned;

    source[0].n_next = 1;
    source[0].next_state = next_state;
    source[0].next_tprob = next_tprob;
    source[1].n_prior = 1;
    source[1].prior_state = prior_state;
    source[1].prior_tprob = prior_tprob;
    next_state[0] = 1;
    next_tprob[0] = 0.75f;
    prior_state[0] = 0;
    prior_tprob[0] = 0.75f;

    owned = pstrain_bw_copy_state_seq(source, 2);
    CHECK(owned != NULL, "copy allocation");
    CHECK(owned != source, "copy owns state array");
    CHECK(owned[0].next_state != source[0].next_state, "copy owns next array");
    CHECK(owned[1].prior_state != source[1].prior_state, "copy owns prior array");
    CHECK(owned[0].next_state[0] == 1 && owned[1].prior_state[0] == 0,
          "copied adjacency values");

    ckd_free(next_state);
    ckd_free(next_tprob);
    ckd_free(prior_state);
    ckd_free(prior_tprob);
    pstrain_bw_free_state_seq(owned, 2);
    printf("PASS: BW state sequence is caller-owned\n");
    return 0;
}
