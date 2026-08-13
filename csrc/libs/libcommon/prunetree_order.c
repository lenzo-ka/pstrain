#include <s3/prunetree_order.h>

#include <assert.h>
#include <string.h>

#define LEFT(i) (((i) << 1) + 1)
#define RIGHT(i) (((i) << 1) + 2)
#define PARENT(i) (((i) - 1) >> 1)

int
prune_candidate_before(const prune_candidate_t *a,
                       const prune_candidate_t *b)
{
    int cmp;

    if (a->gain != b->gain)
        return a->gain < b->gain;

    /* Gain alone is not a total order, so equal-gain candidates used to be
     * selected according to heap insertion history.  Break exact ties by
     * (base-phone name, emitting-state number, root-to-node yes/no path).
     * The name and state are model identities, and the path is fixed by the
     * tree's questions and branch answers.  None depends on allocation,
     * traversal, extraction history, or a container index; the path also
     * distinguishes every pair of nodes within one phone/state tree.
     */
    cmp = strcmp(a->phone, b->phone);
    if (cmp != 0)
        return cmp < 0;
    if (a->state != b->state)
        return a->state < b->state;
    cmp = strcmp(a->path, b->path);
    assert(cmp != 0 || a->value == b->value);
    return cmp < 0;
}

static void
heapify(prune_candidate_t *heap, uint32 pos, uint32 size)
{
    uint32 left = LEFT(pos);
    uint32 right = RIGHT(pos);
    uint32 best = pos;
    prune_candidate_t tmp;

    if (left < size && prune_candidate_before(&heap[left], &heap[best]))
        best = left;
    if (right < size && prune_candidate_before(&heap[right], &heap[best]))
        best = right;
    if (best != pos) {
        tmp = heap[pos];
        heap[pos] = heap[best];
        heap[best] = tmp;
        heapify(heap, best, size);
    }
}

uint32
prune_candidate_insert(prune_candidate_t *heap, uint32 size,
                       prune_candidate_t candidate)
{
    uint32 pos = size++;

    while (pos > 0) {
        uint32 parent = PARENT(pos);
        if (prune_candidate_before(&heap[parent], &candidate))
            break;
        heap[pos] = heap[parent];
        pos = parent;
    }
    heap[pos] = candidate;
    return size;
}

uint32
prune_candidate_extract(prune_candidate_t *out,
                        prune_candidate_t *heap, uint32 size)
{
    assert(size > 0);
    *out = heap[0];
    --size;
    if (size != 0) {
        heap[0] = heap[size];
        heapify(heap, 0, size);
    }
    return size;
}
