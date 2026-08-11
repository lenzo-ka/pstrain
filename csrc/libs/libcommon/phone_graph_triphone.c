/**
 * @file phone_graph_triphone.c
 * @brief Two-sided context cross-product and triphone resolution.
 *
 * Two operations live here:
 *
 *   phone_graph_split_contexts(in) -> out
 *     Build a new graph where every non-filler slot has one unambiguous
 *     (left CI phone, right CI phone) pair. Slots are duplicated over the
 *     cross-product of distinct predecessor and successor contexts, and
 *     edges connect only compatible copies. Fillers stay shared.
 *
 *   cvt2triphone_graph(graph, acmod_set) -> int
 *     Walks an already-unambiguous-context graph and replaces each
 *     slot's CI `phone` with its triphone acmod_id, using the same
 *     fallbacks (word-position back-off, filler handling) as
 *     cvt2triphone() in the linear path.
 *
 * The contract is that callers run split first and cvt2triphone_graph
 * second; both are skipped automatically when there are no
 * multi-pron branches in the input.
 */

#include <s3/phone_graph.h>
#include <s3/acmod_set.h>
#include <s3/s3.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>

#include <assert.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* phone_graph_split_contexts                                         */
/* ------------------------------------------------------------------ */

/* Per-old-slot split metadata. */
typedef struct split_info_s {
    uint32 n_left;
    uint32 n_right;
    uint32 n_copies;
    uint32 first_new;
    acmod_id_t *left_ci;
    acmod_id_t *right_ci;
} split_info_t;

static acmod_id_t
context_phone(acmod_set_t *acmod_set, acmod_id_t phone, acmod_id_t sil)
{
    if (acmod_set_has_attrib(acmod_set, phone, "filler"))
        return sil;
    return acmod_set_base_phone(acmod_set, phone);
}

static acmod_id_t *
distinct_contexts(const phone_graph_t *in,
                  acmod_set_t *acmod_set,
                  const uint32 *neighbors,
                  uint32 n_neighbors,
                  acmod_id_t sil,
                  uint32 *n_distinct)
{
    acmod_id_t *distinct;
    uint32 i, j;

    distinct = ckd_calloc(n_neighbors ? n_neighbors : 1, sizeof(acmod_id_t));
    *n_distinct = 0;
    for (i = 0; i < n_neighbors; ++i) {
        acmod_id_t ci = context_phone(acmod_set, in->phone[neighbors[i]], sil);
        for (j = 0; j < *n_distinct; ++j) {
            if (distinct[j] == ci)
                break;
        }
        if (j == *n_distinct)
            distinct[(*n_distinct)++] = ci;
    }
    if (*n_distinct == 0) {
        distinct[0] = sil;
        *n_distinct = 1;
    }
    return distinct;
}

/* Classify every old slot by its left/right CI-context cross-product. */
static split_info_t *
classify_slots(const phone_graph_t *in,
               acmod_set_t *acmod_set,
               uint32 *total_new)
{
    uint32 i;
    split_info_t *info = ckd_calloc(in->n, sizeof(split_info_t));
    uint32 running = 0;
    acmod_id_t sil = acmod_set_name2id(acmod_set, "SIL");

    for (i = 0; i < in->n; i++) {
        info[i].first_new = running;
        if (acmod_set_has_attrib(acmod_set, in->phone[i], "filler")) {
            /* Fillers remain CI and, crucially, utterance-final SIL remains
             * one shared final HMM.  Its contexts are not model identity. */
            info[i].n_left = info[i].n_right = info[i].n_copies = 1;
            info[i].left_ci = ckd_calloc(1, sizeof(acmod_id_t));
            info[i].right_ci = ckd_calloc(1, sizeof(acmod_id_t));
            info[i].left_ci[0] = info[i].right_ci[0] = sil;
        } else {
            /* A slot's triphone identity depends on both sides.  Variant
             * fan-in changes the left context and variant fan-out changes
             * the right context, so materialize their cross-product. */
            info[i].left_ci = distinct_contexts(in, acmod_set,
                                                in->prior_idx[i], in->n_prior[i],
                                                sil, &info[i].n_left);
            info[i].right_ci = distinct_contexts(in, acmod_set,
                                                 in->next_idx[i], in->n_next[i],
                                                 sil, &info[i].n_right);
            info[i].n_copies = info[i].n_left * info[i].n_right;
        }
        running += info[i].n_copies;
    }

    *total_new = running;
    return info;
}

static void
free_split_info(split_info_t *info, uint32 n)
{
    uint32 i;
    if (!info) return;
    for (i = 0; i < n; i++) {
        if (info[i].left_ci) ckd_free(info[i].left_ci);
        if (info[i].right_ci) ckd_free(info[i].right_ci);
    }
    ckd_free(info);
}

/* Find the deterministic group index for one neighboring CI context. */
static uint32
context_group(const acmod_id_t *contexts, uint32 n, acmod_id_t ci)
{
    uint32 g;
    for (g = 0; g < n; ++g)
        if (contexts[g] == ci)
            return g;
    assert(0 && "context_group: edge context not found");
    return 0;
}

phone_graph_t *
phone_graph_split_contexts(const phone_graph_t *in, acmod_set_t *acmod_set)
{
    uint32 i, u, g, total_new;
    acmod_id_t sil;
    phone_graph_t *out;
    split_info_t *info;

    if (!in || !acmod_set) return NULL;

    info = classify_slots(in, acmod_set, &total_new);
    sil = acmod_set_name2id(acmod_set, "SIL");

    /* Fast path: no slot needs splitting. Make a structural copy of
     * the input so caller-side semantics stay uniform (caller frees
     * both graphs). */
    {
        int any_split = 0;
        for (i = 0; i < in->n; i++) {
            if (info[i].n_copies > 1) { any_split = 1; break; }
        }
        if (!any_split) {
            out = phone_graph_alloc(in->n);
            for (i = 0; i < in->n; i++) {
                out->phone[i]    = in->phone[i];
                out->btw_mark[i] = in->btw_mark[i];
                out->n_next[i]   = in->n_next[i];
                out->n_prior[i]  = in->n_prior[i];
                if (in->n_next[i] > 0) {
                    out->next_idx[i] = ckd_calloc(in->n_next[i], sizeof(uint32));
                    memcpy(out->next_idx[i], in->next_idx[i],
                           in->n_next[i] * sizeof(uint32));
                }
                if (in->n_prior[i] > 0) {
                    out->prior_idx[i] = ckd_calloc(in->n_prior[i], sizeof(uint32));
                    memcpy(out->prior_idx[i], in->prior_idx[i],
                           in->n_prior[i] * sizeof(uint32));
                }
            }
            free_split_info(info, in->n);
            return out;
        }
    }

    /* Allocate output. */
    out = phone_graph_alloc(total_new);

    /* Fill phone[] and btw_mark[] for every copy. */
    for (i = 0; i < in->n; i++) {
        for (g = 0; g < info[i].n_copies; g++) {
            uint32 ns = info[i].first_new + g;
            out->phone[ns]    = in->phone[i];
            out->btw_mark[ns] = in->btw_mark[i];
        }
    }

    /*
     * Adjacency lists are built in three passes (the same
     * count -> allocate -> fill pattern state_seq.c uses), avoiding
     * per-edge realloc in a hot loop:
     *
     *   1. Walk every OLD edge (p -> c) and accumulate per-new-slot
     *      counts. Each old edge becomes n_copies[p] new edges (one
     *      per copy of p, all going into the same c_target derived
     *      from c's split partition).
     *   2. Allocate out->next_idx[i] and out->prior_idx[i] with the
     *      exact sizes from step 1.
     *   3. Walk old edges a second time and fill the arrays using a
     *      per-slot write cursor.
     */
    {
        uint32 *next_cursor;
        uint32 *prior_cursor;

        next_cursor  = ckd_calloc(total_new, sizeof(uint32));
        prior_cursor = ckd_calloc(total_new, sizeof(uint32));

        /* Pass 1: count. */
        for (i = 0; i < in->n; i++) {
            for (u = 0; u < in->n_prior[i]; u++) {
                uint32 p = in->prior_idx[i][u];
                acmod_id_t p_ci = context_phone(acmod_set, in->phone[p], sil);
                acmod_id_t c_ci = context_phone(acmod_set, in->phone[i], sil);
                uint32 p_right = acmod_set_has_attrib(acmod_set, in->phone[p], "filler")
                    ? 0 : context_group(info[p].right_ci, info[p].n_right, c_ci);
                uint32 c_left = acmod_set_has_attrib(acmod_set, in->phone[i], "filler")
                    ? 0 : context_group(info[i].left_ci, info[i].n_left, p_ci);
                uint32 pl, cr;
                for (pl = 0; pl < info[p].n_left; ++pl) {
                    uint32 p_source = info[p].first_new + pl * info[p].n_right + p_right;
                    for (cr = 0; cr < info[i].n_right; ++cr) {
                        uint32 c_target = info[i].first_new + c_left * info[i].n_right + cr;
                        ++out->n_next[p_source];
                        ++out->n_prior[c_target];
                    }
                }
            }
        }

        /* Pass 2: allocate per-slot arrays at exact sizes. */
        for (i = 0; i < total_new; i++) {
            if (out->n_next[i] > 0) {
                out->next_idx[i] = ckd_calloc(out->n_next[i], sizeof(uint32));
            }
            if (out->n_prior[i] > 0) {
                out->prior_idx[i] = ckd_calloc(out->n_prior[i], sizeof(uint32));
            }
        }

        /* Pass 3: fill, advancing per-slot write cursors. */
        for (i = 0; i < in->n; i++) {
            for (u = 0; u < in->n_prior[i]; u++) {
                uint32 p = in->prior_idx[i][u];
                acmod_id_t p_ci = context_phone(acmod_set, in->phone[p], sil);
                acmod_id_t c_ci = context_phone(acmod_set, in->phone[i], sil);
                uint32 p_right = acmod_set_has_attrib(acmod_set, in->phone[p], "filler")
                    ? 0 : context_group(info[p].right_ci, info[p].n_right, c_ci);
                uint32 c_left = acmod_set_has_attrib(acmod_set, in->phone[i], "filler")
                    ? 0 : context_group(info[i].left_ci, info[i].n_left, p_ci);
                uint32 pl, cr;
                for (pl = 0; pl < info[p].n_left; ++pl) {
                    uint32 p_source = info[p].first_new + pl * info[p].n_right + p_right;
                    for (cr = 0; cr < info[i].n_right; ++cr) {
                        uint32 c_target = info[i].first_new + c_left * info[i].n_right + cr;
                        out->next_idx[p_source][next_cursor[p_source]++] = c_target;
                        out->prior_idx[c_target][prior_cursor[c_target]++] = p_source;
                    }
                }
            }
        }

        ckd_free(next_cursor);
        ckd_free(prior_cursor);
    }

    free_split_info(info, in->n);
    return out;
}

/* ------------------------------------------------------------------ */
/* cvt2triphone_graph                                                 */
/* ------------------------------------------------------------------ */

/* Re-implementation of cvt2triphone.c's btw_posn() (which isn't
 * declared in any header). Same semantics: advance the word-position
 * state given the boundary marker of the current phone. */
static word_posn_t
graph_btw_posn(char btw_mark, word_posn_t posn)
{
    if (btw_mark) {
        if (posn == WORD_POSN_INTERNAL || posn == WORD_POSN_BEGIN) {
            return WORD_POSN_END;
        }
        if (posn == WORD_POSN_END) return WORD_POSN_SINGLE;
        if (posn == WORD_POSN_SINGLE) return WORD_POSN_SINGLE;
        E_FATAL("Unhandled word position\n");
    } else {
        if (posn == WORD_POSN_BEGIN) return WORD_POSN_INTERNAL;
        if (posn == WORD_POSN_END || posn == WORD_POSN_SINGLE) {
            return WORD_POSN_BEGIN;
        }
        if (posn == WORD_POSN_INTERNAL) return WORD_POSN_INTERNAL;
        E_FATAL("Unhandled word position\n");
    }
    return posn;
}

int
phone_graph_visit_triphones(const phone_graph_t *graph,
                            acmod_set_t *acmod_set,
                            phone_graph_triphone_visitor_t visitor,
                            void *user_data)
{
    uint32 i;
    acmod_id_t sil;
    word_posn_t *posn_track;

    if (!graph || !acmod_set || !visitor) return S3_ERROR;
    if (graph->n == 0) return S3_SUCCESS;

    sil = acmod_set_name2id(acmod_set, "SIL");
    posn_track = ckd_calloc(graph->n, sizeof(word_posn_t));
    for (i = 0; i < graph->n; ++i) {
        acmod_id_t b = graph->phone[i];
        acmod_id_t l = sil;
        acmod_id_t r = sil;
        word_posn_t in_posn;

        if (graph->n_prior[i] == 0)
            in_posn = WORD_POSN_END;
        else
            in_posn = posn_track[graph->prior_idx[i][0]];
        posn_track[i] = graph_btw_posn(graph->btw_mark[i], in_posn);

        if (graph->n_prior[i] != 0) {
            acmod_id_t phone = graph->phone[graph->prior_idx[i][0]];
            if (!acmod_set_has_attrib(acmod_set, phone, "filler"))
                l = acmod_set_base_phone(acmod_set, phone);
        }
        if (graph->n_next[i] != 0) {
            acmod_id_t phone = graph->phone[graph->next_idx[i][0]];
            if (!acmod_set_has_attrib(acmod_set, phone, "filler"))
                r = acmod_set_base_phone(acmod_set, phone);
        }
        if (!acmod_set_has_attrib(acmod_set, b, "filler") &&
            visitor(acmod_set_base_phone(acmod_set, b), l, r,
                    posn_track[i], user_data) != S3_SUCCESS) {
            ckd_free(posn_track);
            return S3_ERROR;
        }
    }
    ckd_free(posn_track);
    return S3_SUCCESS;
}

int
cvt2triphone_graph(phone_graph_t *graph, acmod_set_t *acmod_set)
{
    uint32 i;
    acmod_id_t sil;
    acmod_id_t *new_phone;
    word_posn_t *posn_track;

    if (!graph || !acmod_set) return S3_ERROR;
    if (graph->n == 0) return S3_SUCCESS;

    if (acmod_set_n_multi(acmod_set) == 0) {
        /* No triphones in the model; nothing to do. Matches
         * cvt2triphone()'s early-out. */
        return S3_SUCCESS;
    }

    sil = acmod_set_name2id(acmod_set, "SIL");

    /* We compute the new triphone id for each slot from the CI ids in
     * `graph->phone[]`, but we MUST NOT overwrite slot i's phone[]
     * before we read it as a left/right context for its neighbors.
     * Use a scratch buffer for the new ids and assign at the end. */
    new_phone = ckd_calloc(graph->n, sizeof(acmod_id_t));

    /* Per-slot word position. Each slot inherits from a predecessor's
     * "outgoing" position state, with the boundary marker advancing
     * it. Context splitting preserves each original slot's word position,
     * so every predecessor produces the same state at slot i. We rely
     * on the slot ordering produced by mk_phone_graph (per-variant
     * sequential, words in transcript order) which is a valid topo
     * order. */
    posn_track = ckd_calloc(graph->n, sizeof(word_posn_t));

    for (i = 0; i < graph->n; i++) {
        word_posn_t in_posn;
        acmod_id_t b, l, r;
        acmod_id_t tri_id;
        int found;
        int j;

        b = graph->phone[i];

        /* Determine incoming word-position state. For slot 0 (no
         * predecessor), seed with WORD_POSN_END as the linear
         * cvt2triphone does. Otherwise inherit from any predecessor;
         * by post-split construction all predecessors agree on the
         * outgoing posn state. */
        if (graph->n_prior[i] == 0) {
            in_posn = WORD_POSN_END;
        } else {
            in_posn = posn_track[graph->prior_idx[i][0]];
        }

        posn_track[i] = graph_btw_posn(graph->btw_mark[i], in_posn);

        /* Left context: the CI phone of any predecessor (after split
         * they all agree). For slot 0 (no predecessor), use SIL. */
        if (graph->n_prior[i] == 0) {
            l = sil;
        } else {
            acmod_id_t pred_phone = graph->phone[graph->prior_idx[i][0]];
            if (acmod_set_has_attrib(acmod_set, pred_phone, "filler")) {
                l = sil;
            } else {
                l = acmod_set_base_phone(acmod_set, pred_phone);
            }
        }

        /* Right context: after context splitting, every successor copy has
         * the same CI center phone. */
        if (graph->n_next[i] == 0) {
            r = sil;
        } else if (graph->n_next[i] == 1) {
            acmod_id_t succ_phone = graph->phone[graph->next_idx[i][0]];
            if (acmod_set_has_attrib(acmod_set, succ_phone, "filler")) {
                r = sil;
            } else {
                r = acmod_set_base_phone(acmod_set, succ_phone);
            }
        } else {
            acmod_id_t succ_phone = graph->phone[graph->next_idx[i][0]];
            if (acmod_set_has_attrib(acmod_set, succ_phone, "filler")) {
                r = sil;
            } else {
                r = acmod_set_base_phone(acmod_set, succ_phone);
            }
        }

        /* If the center phone is a filler, leave the CI id alone
         * (matches the linear path). */
        if (acmod_set_has_attrib(acmod_set, b, "filler")) {
            new_phone[i] = b;
            continue;
        }

        tri_id = acmod_set_tri2id(acmod_set, b, l, r, posn_track[i]);
        if (tri_id != NO_ACMOD) {
            new_phone[i] = tri_id;
            continue;
        }
        /* Back off across word positions, same as the linear path. */
        found = 0;
        for (j = 0; j < N_WORD_POSN; ++j) {
            tri_id = acmod_set_tri2id(acmod_set, b, l, r, j);
            if (tri_id != NO_ACMOD) {
                new_phone[i] = tri_id;
                found = 1;
                break;
            }
        }
        if (!found) {
            /* Leave as CI id; the linear path does the same with an
             * (off-by-default) E_WARN. */
            new_phone[i] = b;
        }
    }

    /* Commit the new ids. */
    for (i = 0; i < graph->n; i++) {
        graph->phone[i] = new_phone[i];
    }

    ckd_free(new_phone);
    ckd_free(posn_track);
    return S3_SUCCESS;
}
