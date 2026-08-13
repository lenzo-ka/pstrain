#include <s3/prunetree_order.h>

#include <assert.h>
#include <string.h>

static void
survivors(const prune_candidate_t *input, const int *order, char out[4])
{
    prune_candidate_t heap[6];
    prune_candidate_t removed;
    int alive[6] = { 1, 1, 1, 1, 1, 1 };
    uint32 size = 0;
    int i;
    int pos = 0;

    for (i = 0; i < 6; ++i)
        size = prune_candidate_insert(heap, size, input[order[i]]);
    for (i = 0; i < 3; ++i) {
        size = prune_candidate_extract(&removed, heap, size);
        alive[(int)(size_t)removed.value] = 0;
    }
    for (i = 0; i < 6; ++i) {
        if (alive[i])
            out[pos++] = input[i].path[0];
    }
    out[pos] = '\0';
}

int
main(void)
{
    prune_candidate_t candidates[6] = {
        { 1.0f, "AH", 2, "f", (void *)(size_t)0 },
        { 1.0f, "AH", 0, "e", (void *)(size_t)1 },
        { 1.0f, "AH", 0, "b", (void *)(size_t)2 },
        { 1.0f, "AA", 1, "a", (void *)(size_t)3 },
        { 2.0f, "AA", 0, "d", (void *)(size_t)4 },
        { 1.0f, "AH", 0, "c", (void *)(size_t)5 },
    };
    int forward[6] = { 0, 1, 2, 3, 4, 5 };
    int shuffled[6] = { 4, 2, 0, 5, 3, 1 };
    char first[4];
    char second[4];

    survivors(candidates, forward, first);
    survivors(candidates, shuffled, second);
    assert(strcmp(first, second) == 0);
    assert(strcmp(first, "fed") == 0);
    return 0;
}
