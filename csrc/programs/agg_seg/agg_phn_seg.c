/* ====================================================================
 * Copyright (c) 1996-2000 Carnegie Mellon University.  All rights
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
 * File: agg_phn_seg.c
 *
 * Description:
 *
 * Author:
 *
 *********************************************************************/

#include "agg_phn_seg.h"
#include "mk_seg.h"
#include "omission.h"

#include <s3/segdmp.h>
#include <s3/corpus.h>
#include <s3/mk_phone_list.h>
#include <s3/ck_seg.h>
#include <s3/cvt2triphone.h>

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/feat.h>

#include <stdio.h>
#include <assert.h>
#include <sys_compat/file.h>


int
agg_phn_seg(lexicon_t *lex,
	    acmod_set_t *acmod_set,
	    feat_t *fcb,
	    segdmp_type_t type)
{
    uint16 *seg;
    vector_t *mfcc;
    vector_t **feat;
    int32 n_frame;
    uint32 tick_cnt;

    acmod_id_t *phone;
    uint32 *start;
    uint32 *len;
    uint32 n_phone;
    uint32 s;
    char *btw_mark;

    char *trans;
    char **word;
    uint32 n_word;
    int32 mfc_veclen = cmd_ln_int32("-ceplen");

    uint32 n_stream;
    uint32 *veclen;

    tick_cnt = 0;

    n_stream = feat_dimension1(fcb);
    veclen = feat_stream_lengths(fcb);

    while (corpus_next_utt()) {
	/* Everything the iteration allocates is released at next_utt, so
	 * every skip below leaves through that one path. */
	trans = NULL;
	seg = NULL;
	word = NULL;
	phone = NULL;
	btw_mark = NULL;
	start = NULL;
	len = NULL;
	mfcc = NULL;
	feat = NULL;

	if ((++tick_cnt % 500) == 0) {
	    E_INFOCONT("[%u] ", tick_cnt);
	}

	if (corpus_get_sent(&trans) != S3_SUCCESS) {
	    E_WARN("Unable to read word transcript for %s; skipping\n", corpus_utt_brief_name());
	    agg_omission_record(AGG_OMIT_TRANSCRIPT_READ);
	    goto next_utt;
	}

	if (corpus_get_seg(&seg, &n_frame) != S3_SUCCESS) {
	    E_WARN("Unable to read Viterbi state segmentation for %s; skipping\n", corpus_utt_brief_name());
	    agg_omission_record(AGG_OMIT_SEGMENTATION_READ);
	    goto next_utt;
	}

	n_word = str2words(trans, NULL, 0);
	word = ckd_calloc(n_word, sizeof(char*));
	str2words(trans, word, n_word);

	phone = mk_phone_list(&btw_mark, &n_phone, word, n_word, lex);
	start = ckd_calloc(n_phone, sizeof(uint32));
	len = ckd_calloc(n_phone, sizeof(uint32));

	/* check to see whether the word transcript and dictionary entries
	   agree with the state segmentation */
	if (ck_seg(acmod_set, phone, n_phone, seg, n_frame, corpus_utt()) != S3_SUCCESS) {
	    E_ERROR("ck_seg failed");
	    agg_omission_record(AGG_OMIT_SEGMENTATION_MISMATCH);

	    goto next_utt;
	}

	if (cvt2triphone(acmod_set, phone, btw_mark, n_phone) != S3_SUCCESS) {
	    E_ERROR("cvt2triphone failed");
	    agg_omission_record(AGG_OMIT_TRIPHONE_CONVERSION);

	    goto next_utt;
	}

	if (mk_seg(acmod_set,
		   seg,
		   n_frame,
		   phone,
		   start,
		   len,
		   n_phone) != S3_SUCCESS) {
	    E_ERROR("mk_seg failed");
	    agg_omission_record(AGG_OMIT_SEGMENT_GENERATION);
	    goto next_utt;
	}

	if (!corpus_provides_mfcc()) {
	    E_FATAL("No data type specified\n");
	}

	if (sys_compat_access(corpus_mfcc_filename(), R_OK) != 0) {
	    E_WARN("Can't read input features from %s; skipping\n", corpus_utt());
	    agg_omission_record(AGG_OMIT_FEATURE_READ);
	    goto next_utt;
	}

	if (corpus_get_generic_featurevec(&mfcc, &n_frame, mfc_veclen) < 0) {
	    E_FATAL("Can't read input features from %s\n", corpus_utt());
	}

	if (n_frame < 9) {
	    E_WARN("utt %s too short\n", corpus_utt());
	    agg_omission_record(AGG_OMIT_TOO_SHORT);
	    goto next_utt;
	}

	feat = feat_array_alloc(fcb, n_frame + feat_window_size(fcb));
	feat_s2mfc2feat_live(fcb, mfcc, &n_frame, TRUE, TRUE, feat);

	for (s = 0; s < n_phone; s++) {
	    segdmp_add_feat(phone[s],
			    &feat[start[s]],
			    len[s]);
	}

	agg_omission_processed();

    next_utt:
	if (feat)
	    feat_array_free(feat);
	if (mfcc) {
	    free(mfcc[0]);	/* alloc'ed using malloc in areadfloat() */
	    ckd_free(mfcc);
	}
	if (trans)
	    free(trans);	/* alloc'ed using strdup, not ckd_*() */
	if (seg)
	    free(seg);		/* alloc'ed using malloc in areadshort(), not ckd_*() */
	if (word)
	    ckd_free(word);
	if (phone)
	    ckd_free(phone);
	if (btw_mark)
	    ckd_free(btw_mark);
	if (start)
	    ckd_free(start);
	if (len)
	    ckd_free(len);
    }

    return 0;
}
