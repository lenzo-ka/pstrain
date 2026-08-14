/* ====================================================================
 * Copyright (c) 1995-2000 Carnegie Mellon University.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * This work was supported in part by funding from the Defense Advanced
 * Research Projects Agency and the National Science Foundation of the
 * United States of America, and the CMU Sphinx Speech Consortium.
 *
 * THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
 * NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ====================================================================
 *
 */
/*********************************************************************
 *
 * File: next_utt_states.c
 *
 * Description:
 * 	Get the word transcript for the next utterance and convert it
 *	into a sequence of states ready for forward/backward.
 *
 * Author:
 * 	Eric H. Thayer, eht@cs.cmu.edu
 *********************************************************************/

#include <s3/lexicon.h>
#include <s3/model_inventory.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>
#include <sphinxbase/strfuncs.h>
#include <s3/mk_phone_list.h>
#include <s3/cvt2triphone.h>
#include <s3/phone_graph.h>
#include <s3/state_seq_graph.h>

#include <s3/state_seq.h>

#include <string.h>

#include "next_utt_states.h"

int
next_utt_states_graph_built(int multipron, int optional_final_silence)
{
    return multipron || optional_final_silence;
}

/* PSTRAIN DIVERGENCE: retain the final SIL HMM but add direct lexical-exit
 * arcs to its non-emitting exit as normalized competing alternatives. */
static state_t *
add_boundary_bypass(state_t *old, uint32 n_state,
                    const phone_graph_t *graph, model_def_t *mdef)
{
    state_t *state;
    uint32 *next_state, *prior_state;
    float32 *next_tprob, *prior_tprob;
    uint32 *offset;
    uint8 *bypass_pred;
    uint32 i, total_next = 0, total_prior = 0, noff = 0, poff = 0;
    uint32 final_slot, final_exit, n_bypass;

    if (graph->n < 3)
        return old;
    offset = ckd_calloc(graph->n, sizeof(*offset));
    bypass_pred = ckd_calloc(n_state, sizeof(*bypass_pred));
    for (i = 1; i < graph->n; ++i)
        offset[i] = offset[i - 1] + mdef->defn[graph->phone[i - 1]].n_state;
    final_slot = graph->n - 1;
    final_exit = offset[final_slot] + mdef->defn[graph->phone[final_slot]].n_state - 1;
    n_bypass = graph->n_prior[final_slot];
    for (i = 0; i < n_bypass; ++i) {
        uint32 pred = graph->prior_idx[final_slot][i];
        uint32 pred_exit = offset[pred] + mdef->defn[graph->phone[pred]].n_state - 1;
        bypass_pred[pred_exit] = TRUE;
    }

    for (i = 0; i < n_state; ++i) {
        total_next += old[i].n_next;
        total_prior += old[i].n_prior;
    }
    total_next += n_bypass;
    total_prior += n_bypass;
    state = ckd_calloc(n_state, sizeof(*state));
    next_state = ckd_calloc(total_next, sizeof(*next_state));
    next_tprob = ckd_calloc(total_next, sizeof(*next_tprob));
    prior_state = ckd_calloc(total_prior, sizeof(*prior_state));
    prior_tprob = ckd_calloc(total_prior, sizeof(*prior_tprob));

    for (i = 0; i < n_state; ++i) {
        state[i] = old[i];
        state[i].next_state = next_state + noff;
        state[i].next_tprob = next_tprob + noff;
        if (old[i].n_next) {
            memcpy(state[i].next_state, old[i].next_state,
                   old[i].n_next * sizeof(*next_state));
            memcpy(state[i].next_tprob, old[i].next_tprob,
                   old[i].n_next * sizeof(*next_tprob));
            if (bypass_pred[i]) {
                uint32 slot;
                for (slot = 0; slot < old[i].n_next; ++slot)
                    state[i].next_tprob[slot] *= 0.5f;
            }
        }
        noff += old[i].n_next;
        if (bypass_pred[i]) {
            state[i].next_state[old[i].n_next] = final_exit;
            state[i].next_tprob[old[i].n_next] = 0.5f;
            state[i].n_next++;
            ++noff;
        }

        state[i].prior_state = prior_state + poff;
        state[i].prior_tprob = prior_tprob + poff;
        if (old[i].n_prior) {
            memcpy(state[i].prior_state, old[i].prior_state,
                   old[i].n_prior * sizeof(*prior_state));
            memcpy(state[i].prior_tprob, old[i].prior_tprob,
                   old[i].n_prior * sizeof(*prior_tprob));
            {
                uint32 slot;
                for (slot = 0; slot < old[i].n_prior; ++slot) {
                    if (bypass_pred[old[i].prior_state[slot]])
                        state[i].prior_tprob[slot] *= 0.5f;
                }
            }
        }
        poff += old[i].n_prior;
        if (i == final_exit) {
            uint32 slot;
            for (slot = 0; slot < n_bypass; ++slot) {
                uint32 pred = graph->prior_idx[final_slot][slot];
                state[i].prior_state[old[i].n_prior + slot] =
                    offset[pred] + mdef->defn[graph->phone[pred]].n_state - 1;
                state[i].prior_tprob[old[i].n_prior + slot] = 0.5f;
            }
            state[i].n_prior += n_bypass;
            poff += n_bypass;
        }
    }

    state_seq_free(old, n_state);
    ckd_free(offset);
    ckd_free(bypass_pred);
    return state;
}

state_t *next_utt_states(uint32 *n_state,
			 lexicon_t *lex,
			 model_inventory_t *inv,
			 model_def_t *mdef,
			 char *trans,
			 int optional_final_silence
			 )
{
    char **word;
    char *utterance;
    uint32 n_word;
    uint32 n_phone;
    char *btw_mark;
    acmod_set_t *acmod_set;
    acmod_id_t *phone;

    state_t *state_seq;

    utterance = ckd_salloc(trans);
    n_word = str2words(utterance, NULL, 0);
    word = ckd_calloc(n_word, sizeof(char*));
    str2words(utterance, word, n_word);

    phone = mk_phone_list(&btw_mark, &n_phone, word, n_word, lex);

    if (phone == NULL) {
	E_WARN("Unable to produce phonetic transcription for the utterance '%s'\n", trans);
	ckd_free(word);
	ckd_free(utterance);
	return NULL;
    }

    acmod_set = inv->acmod_set;

#ifdef NEXT_UTT_STATES_VERBOSE
    print_phone_list(phone, n_phone, btw_mark, acmod_set);
#endif

    cvt2triphone(acmod_set, phone, btw_mark, n_phone);

#ifdef NEXT_UTT_STATES_VERBOSE
    print_phone_list(phone, n_phone, btw_mark, acmod_set);
#endif

    state_seq = state_seq_make(n_state, phone, n_phone, inv, mdef);

#ifdef NEXT_UTT_STATES_VERBOSE
    state_seq_print(state_seq, *n_state, mdef);
#endif

    ckd_free(phone);
    ckd_free(btw_mark);
    ckd_free(word);
    ckd_free(utterance);

    return state_seq;
}

state_t *next_utt_states_graph(uint32 *n_state,
			       lexicon_t *lex,
			       model_inventory_t *inv,
			       model_def_t *mdef,
			       char *trans,
			       int multipron,
			       int optional_final_silence
			       )
{
    char *utterance;
    char **word;
    uint32 n_word;
    phone_graph_t *graph;
    phone_graph_t *split;
    state_t *state_seq;

    /* str2words mutates its input; work on a private copy so we leave
     * `trans` untouched for any caller that wants to log it after. */
    utterance = ckd_salloc(trans);
    n_word = str2words(utterance, NULL, 0);
    if (n_word == 0) {
	E_WARN("Empty transcript\n");
	ckd_free(utterance);
	return NULL;
    }
    word = ckd_calloc(n_word, sizeof(char *));
    str2words(utterance, word, n_word);

    graph = mk_phone_graph(word, n_word, lex, multipron);
    ckd_free(word);
    ckd_free(utterance);
    if (graph == NULL) {
	/* mk_phone_graph has already logged the offending word. */
	return NULL;
    }

    split = phone_graph_split_contexts(graph, inv->acmod_set);
    phone_graph_free(graph);
    if (split == NULL) {
	E_ERROR("phone_graph_split_contexts failed\n");
	return NULL;
    }

    if (cvt2triphone_graph(split, inv->acmod_set) != S3_SUCCESS) {
	E_ERROR("cvt2triphone_graph failed\n");
	phone_graph_free(split);
	return NULL;
    }

    state_seq = state_seq_make_graph(n_state, split, inv, mdef);
    if (state_seq == NULL) {
	E_ERROR("state_seq_make_graph failed\n");
	phone_graph_free(split);
	return NULL;
    }

    if (optional_final_silence)
        state_seq = add_boundary_bypass(state_seq, *n_state, split, mdef);
    phone_graph_free(split);

    return state_seq;
}

state_t *next_utt_states_mmie(uint32 *n_state,
			      lexicon_t *lex,
			      model_inventory_t *inv,
			      model_def_t *mdef,
			      char *curr_word,
			      acmod_id_t *l_phone,
			      acmod_id_t *r_phone
			      )
{
  uint32 n_phone;
  char *btw_mark;
  acmod_set_t *acmod_set;
  acmod_id_t *phone;

  state_t *state_seq;

  phone = mk_word_phone_list(&btw_mark, &n_phone, curr_word,lex);

  if (phone == NULL) {
    E_WARN("Unable to produce phonetic transcription for the word '%s'\n", curr_word);
    return NULL;
  }

  acmod_set = inv->acmod_set;

  cvt2triphone_mmie(acmod_set, phone, l_phone, r_phone, btw_mark, n_phone);

  state_seq = state_seq_make(n_state, phone, n_phone, inv, mdef);

  ckd_free(phone);
  ckd_free(btw_mark);

  return state_seq;
}
