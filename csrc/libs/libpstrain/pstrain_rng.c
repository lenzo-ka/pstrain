#include <pstrain/rng.h>

#define PSTRAIN_RNG_MASK UINT64_C(0xffffffffffff)
#define PSTRAIN_RNG_MULTIPLIER UINT64_C(0x5deece66d)
#define PSTRAIN_RNG_INCREMENT UINT64_C(0xb)
#define PSTRAIN_RNG_DEFAULT_STATE UINT64_C(0x1234abcd330e)

static uint64_t
pstrain_rng_next(pstrain_rng_t *rng)
{
    rng->state = (PSTRAIN_RNG_MULTIPLIER * rng->state +
                  PSTRAIN_RNG_INCREMENT) & PSTRAIN_RNG_MASK;
    return rng->state;
}

void
pstrain_rng_init(pstrain_rng_t *rng)
{
    rng->state = PSTRAIN_RNG_DEFAULT_STATE;
}

void
pstrain_srand48(pstrain_rng_t *rng, int32_t seed)
{
    rng->state = ((uint64_t)(uint32_t)seed << 16) | UINT64_C(0x330e);
}

double
pstrain_drand48(pstrain_rng_t *rng)
{
    return (double)pstrain_rng_next(rng) * 0x1p-48;
}

int32_t
pstrain_lrand48(pstrain_rng_t *rng)
{
    return (int32_t)(pstrain_rng_next(rng) >> 17);
}
