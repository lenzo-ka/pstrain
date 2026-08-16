#include <pstrain/rng.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
check_libc(void)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
    return 0;
#else
    pstrain_rng_t rng;
    int i;

    pstrain_srand48(&rng, -123456789);
    srand48(-123456789L);
    for (i = 0; i < 100000; ++i) {
        if (pstrain_drand48(&rng) != drand48()) {
            fprintf(stderr, "libc cross-check draw %d differs\n", i);
            return 1;
        }
    }
    return 0;
#endif
}

int
main(void)
{
    return check_golden() || check_libc();
}
