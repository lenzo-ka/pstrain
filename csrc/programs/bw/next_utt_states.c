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

/* Add one shared non-emitting sentence exit when removing </s> exposes
 * multiple terminal pronunciation branches.  BW requires n_state - 1 to be
 * the sole final state, but this state consumes no frame and therefore keeps
 * the boundary bypass exact. */
static state_t *
add_shared_terminal(state_t *old, uint32 *n_state)
{
    state_t *state;
    uint32 *next_state, *prior_state;
    float32 *next_tprob, *prior_tprob;
    uint32 i, terminal_count = 0, total_next = 0, total_prior = 0;
    uint32 noff = 0, poff = 0, final = *n_state;

    for (i = 0; i < *n_state; ++i) {
        total_next += old[i].n_next;
        total_prior += old[i].n_prior;
        if (old[i].n_next == 0)
            ++terminal_count;
    }
    if (terminal_count <= 1)
        return old;

    total_next += terminal_count;
    total_prior += terminal_count;
    state = ckd_calloc(*n_state + 1, sizeof(*state));
    next_state = ckd_calloc(total_next, sizeof(*next_state));
    next_tprob = ckd_calloc(total_next, sizeof(*next_tprob));
    prior_state = ckd_calloc(total_prior, sizeof(*prior_state));
    prior_tprob = ckd_calloc(total_prior, sizeof(*prior_tprob));

    for (i = 0; i < *n_state; ++i) {
        state[i] = old[i];
        state[i].next_state = next_state + noff;
        state[i].next_tprob = next_tprob + noff;
        if (old[i].n_next) {
            memcpy(state[i].next_state, old[i].next_state,
                   old[i].n_next * sizeof(*next_state));
            memcpy(state[i].next_tprob, old[i].next_tprob,
                   old[i].n_next * sizeof(*next_tprob));
        }
        noff += old[i].n_next;
        if (old[i].n_next == 0) {
            state[i].next_state[0] = final;
            state[i].next_tprob[0] = 1.0f;
            state[i].n_next = 1;
            ++noff;
        }

        state[i].prior_state = prior_state + poff;
        state[i].prior_tprob = prior_tprob + poff;
        if (old[i].n_prior) {
            memcpy(state[i].prior_state, old[i].prior_state,
                   old[i].n_prior * sizeof(*prior_state));
            memcpy(state[i].prior_tprob, old[i].prior_tprob,
                   old[i].n_prior * sizeof(*prior_tprob));
        }
        poff += old[i].n_prior;
    }

    state[final] = old[*n_state - 1];
    state[final].mixw = state[final].ci_mixw = TYING_NO_ID;
    state[final].l_mixw = state[final].l_ci_mixw = TYING_NO_ID;
    state[final].cb = state[final].ci_cb = TYING_NO_ID;
    state[final].l_cb = state[final].l_ci_cb = TYING_NO_ID;
    state[final].n_next = 0;
    state[final].next_state = NULL;
    state[final].next_tprob = NULL;
    state[final].n_prior = terminal_count;
    state[final].prior_state = prior_state + poff;
    state[final].prior_tprob = prior_tprob + poff;
    {
        uint32 q = 0;
        for (i = 0; i < *n_state; ++i) {
        if (old[i].n_next == 0) {
                state[final].prior_state[q] = i;
                state[final].prior_tprob[q] = 1.0f;
                ++q;
            }
        }
    }

    state_seq_free(old, *n_state);
    ++*n_state;
    return state;
}

state_t *next_utt_states(uint32 *n_state,
			 lexicon_t *lex,
			 model_inventory_t *inv,
			 model_def_t *mdef,
			 char *trans,
			 int optional_boundary_silence
			 )
{
    char **word;
    char **all_word;
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
    all_word = word;
    str2words(utterance, word, n_word);

    /* PSTRAIN DIVERGENCE: upstream makes transcript <s>/</s> SIL HMMs
     * mandatory.  The historical linear state sequence has no epsilon entry
     * node, so its exact zero-frame bypass is represented by omitting only
     * those explicit boundary words. */
    if (optional_boundary_silence && n_word > 0 && strcmp(word[0], "<s>") == 0) {
        ++word;
        --n_word;
    }
    if (optional_boundary_silence && n_word > 0 && strcmp(word[n_word - 1], "</s>") == 0)
        --n_word;

    phone = mk_phone_list(&btw_mark, &n_phone, word, n_word, lex);

    if (phone == NULL) {
	E_WARN("Unable to produce phonetic transcription for the utterance '%s'\n", trans);
	ckd_free(all_word);
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
    ckd_free(all_word);
    ckd_free(utterance);

    return state_seq;
}

state_t *next_utt_states_graph(uint32 *n_state,
			       lexicon_t *lex,
			       model_inventory_t *inv,
			       model_def_t *mdef,
			       char *trans,
			       int optional_boundary_silence
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

    /* See the linear builder above: the graph-state engine also has one
     * hard-wired entry and exit, so omission is its exact boundary bypass. */
    {
        char **all_word = word;
        if (optional_boundary_silence && n_word > 0 && strcmp(word[0], "<s>") == 0) {
            ++word;
            --n_word;
        }
        if (optional_boundary_silence && n_word > 0 && strcmp(word[n_word - 1], "</s>") == 0)
            --n_word;

        graph = mk_phone_graph(word, n_word, lex, /*multipron=*/ 1);
        ckd_free(all_word);
    }
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
    phone_graph_free(split);
    if (state_seq == NULL) {
	E_ERROR("state_seq_make_graph failed\n");
	return NULL;
    }

    if (optional_boundary_silence)
        state_seq = add_shared_terminal(state_seq, n_state);

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
