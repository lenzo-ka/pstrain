#ifndef PSTRAIN_PRUNETREE_ORDER_H
#define PSTRAIN_PRUNETREE_ORDER_H

#include <s3/s3.h>

typedef struct {
    float32 gain;
    const char *phone;
    uint32 state;
    const char *path;
    void *value;
} prune_candidate_t;

int prune_candidate_before(const prune_candidate_t *a,
                           const prune_candidate_t *b);
uint32 prune_candidate_insert(prune_candidate_t *heap, uint32 size,
                              prune_candidate_t candidate);
uint32 prune_candidate_extract(prune_candidate_t *out,
                               prune_candidate_t *heap, uint32 size);

#endif
