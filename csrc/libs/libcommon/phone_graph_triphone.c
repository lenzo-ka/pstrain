/**
 * @file phone_graph_triphone.c
 * @brief Context-split + triphone resolution on a phone_graph_t.
 *
 * Two operations live here:
 *
 *   phone_graph_split_contexts(in) -> out
 *     Build a new graph where every slot has at most one distinct
 *     CI-phone predecessor. Slots whose predecessors carry multiple
 *     CI phones are duplicated, one copy per distinct predecessor
 *     CI phone, and the predecessor edges are partitioned. All copies
 *     of a slot share the same outgoing-edge set, with each outgoing
 *     edge possibly retargeted via the same split rule applied to the
 *     successor.
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
    uint32     n_copies;     /* >=1; equals n distinct CI predecessor phones */
    uint32     first_new;    /* offset in the new graph */
    acmod_id_t *group_ci;    /* [n_copies] distinct CI phones, in order */
} split_info_t;

/* Classify every old slot: how many copies, which CI phone each copy
 * is for. */
static split_info_t *
classify_slots(const phone_graph_t *in,
               acmod_set_t *acmod_set,
               uint32 *total_new)
{
    uint32 i, u, v;
    split_info_t *info = ckd_calloc(in->n, sizeof(split_info_t));
    uint32 running = 0;

    for (i = 0; i < in->n; i++) {
        /* Fillers are deliberately left as CI models by
         * cvt2triphone_graph(), so their left context is irrelevant.
         * In particular, splitting utterance-final </s> (SIL) creates
         * several terminal HMM exits while forward/backward has exactly
         * one designated final state (n_state - 1). Keep one filler slot
         * so every incoming pronunciation path reaches that shared final.
         * Transition weights and beam thresholds are unchanged. The shared
         * fan-in intentionally changes path/pruning structure: predecessor
         * alpha mass is summed before pruning, preserving pronunciation
         * posterior summation at the single terminal state. */
        if (in->n_prior[i] <= 1
            || acmod_set_has_attrib(acmod_set, in->phone[i], "filler")) {
            info[i].n_copies = 1;
            info[i].group_ci = ckd_calloc(1, sizeof(acmod_id_t));
            info[i].group_ci[0] = (in->n_prior[i] == 1)
                                       ? in->phone[in->prior_idx[i][0]]
                                       : 0;  /* unused for n_copies == 1 */
            info[i].first_new = running;
            running += 1;
            continue;
        }
        /* Collect distinct predecessor CI phones, preserving the
         * order of first occurrence (deterministic for parity). */
        acmod_id_t *distinct = ckd_calloc(in->n_prior[i], sizeof(acmod_id_t));
        uint32 n_distinct = 0;
        for (u = 0; u < in->n_prior[i]; u++) {
            acmod_id_t ci = in->phone[in->prior_idx[i][u]];
            int found = 0;
            for (v = 0; v < n_distinct; v++) {
                if (distinct[v] == ci) { found = 1; break; }
            }
            if (!found) distinct[n_distinct++] = ci;
        }
        info[i].n_copies = n_distinct;
        info[i].group_ci = distinct;     /* hand over ownership */
        info[i].first_new = running;
        running += n_distinct;
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
        if (info[i].group_ci) ckd_free(info[i].group_ci);
    }
    ckd_free(info);
}

/* Find which copy of slot `c` corresponds to predecessor CI phone
 * `pred_ci`. */
static uint32
copy_for_predecessor_ci(const split_info_t *info, uint32 c, acmod_id_t pred_ci)
{
    uint32 g;
    if (info[c].n_copies == 1) {
        return info[c].first_new;
    }
    for (g = 0; g < info[c].n_copies; g++) {
        if (info[c].group_ci[g] == pred_ci) {
            return info[c].first_new + g;
        }
    }
    /* Should be unreachable if the split was computed correctly. */
    assert(0 && "copy_for_predecessor_ci: predecessor CI not found");
    return 0;
}

phone_graph_t *
phone_graph_split_contexts(const phone_graph_t *in, acmod_set_t *acmod_set)
{
    uint32 i, u, g, total_new;
    phone_graph_t *out;
    split_info_t *info;

    if (!in || !acmod_set) return NULL;

    info = classify_slots(in, acmod_set, &total_new);

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
                acmod_id_t pred_ci = in->phone[p];
                uint32 c_target = copy_for_predecessor_ci(info, i, pred_ci);
                uint32 pg;
                for (pg = 0; pg < info[p].n_copies; pg++) {
                    uint32 p_source = info[p].first_new + pg;
                    ++out->n_next[p_source];
                    ++out->n_prior[c_target];
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
                acmod_id_t pred_ci = in->phone[p];
                uint32 c_target = copy_for_predecessor_ci(info, i, pred_ci);
                uint32 pg;
                for (pg = 0; pg < info[p].n_copies; pg++) {
                    uint32 p_source = info[p].first_new + pg;
                    out->next_idx[p_source][next_cursor[p_source]++] = c_target;
                    out->prior_idx[c_target][prior_cursor[c_target]++] = p_source;
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
     * it. Since the graph may have fan-in (all incoming arcs with the
     * same CI predecessor, by construction of split_contexts), every
     * predecessor produces the same word-position state at slot i —
     * so we can compute posn per slot in topological order. We rely
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

        /* Right context: CI phone of any successor (the linear path
         * uses phone[i+1]; for graph, all successors share the same
         * CI center phone because they're variants of the next word's
         * first phone — but wait, they DON'T: variants of word i+1
         * start with potentially different phones. We don't split on
         * the SUCCESSOR side here.) */
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
            /* Multiple successors with possibly different CI phones.
             * This slot is the last phone of a variant of word i, and
             * the next word has multiple variants whose first phones
             * may differ. We cannot fold those into a single triphone
             * id without further splitting on the right context too.
             *
             * For now: use the FIRST successor's CI phone as the
             * right context. This is an approximation that mirrors
             * what the linear cvt2triphone path would do given the
             * pron[1]-only behavior. A future refinement is to also
             * split on right context. */
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
