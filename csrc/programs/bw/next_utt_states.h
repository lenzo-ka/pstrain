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
 * File: next_utt_states.h
 *
 * Description:
 *
 * Author:
 * 	Eric H. Thayer (eht@cs.cmu.edu)
 *********************************************************************/

#ifndef NEXT_UTT_STATES_H
#define NEXT_UTT_STATES_H

#include <s3/state.h>
#include <sphinxbase/prim_type.h>
#include <s3/lexicon.h>
#include <s3/model_inventory.h>
#include <s3/model_def.h>

int next_utt_states_graph_built(int multipron,
                                int optional_boundary_silence);
float32 next_utt_states_initial_tprob(const state_t *state_seq, uint32 i);

state_t *next_utt_states(uint32 *n_state,
			 lexicon_t *lex,
			 model_inventory_t *inv,
			 model_def_t *mdef,
			 char *transcript,
			 int optional_boundary_silence);

/*
 * Graph-aware twin of next_utt_states.
 *
 * Composes mk_phone_graph -> phone_graph_split_contexts ->
 * cvt2triphone_graph -> state_seq_make_graph, so every pronunciation
 * variant of every word contributes a parallel path through the
 * sentence HMM. Use this when multi-pronunciation Baum-Welch is on.
 *
 * Unlike next_utt_states, the returned state_t array is freshly
 * allocated per call (not backed by static buffers inside
 * state_seq_make). The caller is responsible for releasing it with
 * state_seq_free(state_seq, *n_state) when done.
 *
 * Returns NULL on failure (transcript empty, word lookup failed,
 * context-split failed, etc.); an E_WARN/E_ERROR has already been
 * emitted in that case.
 */
state_t *next_utt_states_graph(uint32 *n_state,
			       lexicon_t *lex,
			       model_inventory_t *inv,
			       model_def_t *mdef,
			       char *transcript,
			       int multipron,
			       int optional_boundary_silence);

state_t *next_utt_states_mmie(uint32 *n_state,
			      lexicon_t *lex,
			      model_inventory_t *inv,
			      model_def_t *mdef,
			      char *curr_word,
			      acmod_id_t *l_phone,
			      acmod_id_t *r_phone);

#endif /* NEXT_UTT_STATES_H */
