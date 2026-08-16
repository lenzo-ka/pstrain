#include <pstrain/rng.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_SEQUENCE_LENGTH 100000

static int
check_golden(void)
{
    static const uint64_t states[] = {
        UINT64_C(0x2bbb62dc5101), UINT64_C(0xbff993816378),
        UINT64_C(0x18abd0152a23), UINT64_C(0xded6cf2262f2),
        UINT64_C(0x93ca29a18d15), UINT64_C(0xc92a23e5effc),
        UINT64_C(0xb133a2d39657), UINT64_C(0x5e67775d2d16),
        UINT64_C(0xdfb82d75f669), UINT64_C(0xbebe8d6840c0)
    };
    pstrain_rng_t rng;
    size_t i;

    pstrain_srand48(&rng, 0);
    for (i = 0; i < sizeof(states) / sizeof(states[0]); ++i) {
        double actual = pstrain_drand48(&rng);
        double expected = (double)states[i] * 0x1p-48;
        if (rng.state != states[i] || actual != expected) {
            fprintf(stderr, "golden draw %zu differs\n", i);
            return 1;
        }
    }
    pstrain_rng_init(&rng);
    if (rng.state != UINT64_C(0x1234abcd330e)) {
        fputs("POSIX default state differs\n", stderr);
        return 1;
    }
    return 0;
}

static int
check_continuation(void)
{
    pstrain_rng_t rng;
    float continued;
    float restarted;
    int i;

    /* State 1 consumes five trials times two densities.  State 2 resumes. */
    pstrain_rng_init(&rng);
    for (i = 0; i < 10; ++i)
        (void)pstrain_drand48(&rng);
    continued = (float)pstrain_drand48(&rng);

    pstrain_rng_init(&rng);
    restarted = (float)pstrain_drand48(&rng);
    if ((uint32_t)(continued * 1000) != 691 ||
        (uint32_t)(restarted * 1000) != 396 ||
        continued != 0.691004395f || restarted != 0.396464765f) {
        fputs("default-stream continuation anchor differs\n", stderr);
        return 1;
    }
    return 0;
}

#if !defined(_WIN32) || defined(__CYGWIN__)
static uint64_t
state_from_seed48(const unsigned short seed[3])
{
    return (uint64_t)seed[0] | ((uint64_t)seed[1] << 16) |
           ((uint64_t)seed[2] << 32);
}

static int
check_seeded_drand48(int32_t seed)
{
    pstrain_rng_t rng;
    int i;

    pstrain_srand48(&rng, seed);
    srand48((long)seed);
    for (i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        if (pstrain_drand48(&rng) != drand48()) {
            fprintf(stderr, "seed %d drand48 draw %d differs\n", seed, i);
            return 1;
        }
    }
    return 0;
}

static int
check_lrand48(void)
{
    pstrain_rng_t rng;
    int i;

    pstrain_srand48(&rng, -123456789);
    srand48(-123456789L);
    for (i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        if (pstrain_lrand48(&rng) != lrand48()) {
            fprintf(stderr, "lrand48 draw %d differs\n", i);
            return 1;
        }
    }
    return 0;
}

static int
check_mixed(void)
{
    pstrain_rng_t rng;
    int i;

    pstrain_srand48(&rng, 0x76543210);
    srand48(0x76543210L);
    for (i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        if ((i & 1) == 0) {
            if (pstrain_drand48(&rng) != drand48()) {
                fprintf(stderr, "mixed drand48 draw %d differs\n", i);
                return 1;
            }
        }
        else if (pstrain_lrand48(&rng) != lrand48()) {
            fprintf(stderr, "mixed lrand48 draw %d differs\n", i);
            return 1;
        }
    }
    return 0;
}

static int
check_default(void)
{
    pstrain_rng_t rng;
    int i;

    pstrain_rng_init(&rng);
    for (i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        if (pstrain_drand48(&rng) != drand48()) {
            fprintf(stderr, "default drand48 draw %d differs\n", i);
            return 1;
        }
    }
    return 0;
}

static int
check_seed48_state(const unsigned short initial[3], const char *name)
{
    unsigned short libc_state[3] = {initial[0], initial[1], initial[2]};
    pstrain_rng_t rng;
    int i;

    rng.state = state_from_seed48(initial);
    (void)seed48(libc_state);
    for (i = 0; i < TEST_SEQUENCE_LENGTH; ++i) {
        if ((i & 1) == 0) {
            if (pstrain_drand48(&rng) != drand48()) {
                fprintf(stderr, "%s drand48 draw %d differs\n", name, i);
                return 1;
            }
        }
        else if (pstrain_lrand48(&rng) != lrand48()) {
            fprintf(stderr, "%s lrand48 draw %d differs\n", name, i);
            return 1;
        }
    }
    return 0;
}
#endif

static int
check_libc(void)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
    return 0;
#else
    static const unsigned short zero_state[3] = {0, 0, 0};
    static const unsigned short max_state[3] = {0xffff, 0xffff, 0xffff};

    return check_default() ||
           check_seeded_drand48(-123456789) ||
           check_seeded_drand48(INT32_MIN) ||
           check_seeded_drand48(INT32_MAX) ||
           check_lrand48() ||
           check_mixed() ||
           check_seed48_state(zero_state, "seed48 zero") ||
           check_seed48_state(max_state, "seed48 0xffffffffffff");
#endif
}

int
main(void)
{
    return check_golden() || check_continuation() || check_libc();
}
