#ifndef PSTRAIN_RNG_H
#define PSTRAIN_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pstrain_rng_s {
    uint64_t state;
} pstrain_rng_t;

void pstrain_rng_init(pstrain_rng_t *rng);
void pstrain_srand48(pstrain_rng_t *rng, int32_t seed);
double pstrain_drand48(pstrain_rng_t *rng);
int32_t pstrain_lrand48(pstrain_rng_t *rng);

#ifdef __cplusplus
}
#endif

#endif
