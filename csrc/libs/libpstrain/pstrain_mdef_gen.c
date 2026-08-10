/* ====================================================================
 * Copyright (c) 1999-2016 Carnegie Mellon University.  All rights
 * reserved.
 *
 * This file is derived from CMU SphinxTrain/SphinxBase sources.
 * Modifications for pstrain are Copyright (c) 2026 Kevin Lenzo and are
 * licensed under the BSD 2-Clause license (see LICENSE at the repository
 * root). The repository history records the modifications.
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
 * ====================================================================
 */
/**
 * @file pstrain_mdef_gen.c
 * @brief CFFI-friendly wrappers for mk_mdef_gen functionality.
 */

#include "pstrain_mdef_gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>
#include <sphinxbase/pio.h>

#include <s3/s3.h>
#include <s3/model_def_io.h>
#include <s3/acmod_set.h>

/* We need to include the mk_mdef_gen helper headers */
/* These are in programs/mk_mdef_gen but we'll link against them */

/* Forward declarations for mk_mdef_gen functions */
typedef struct heapelement_s heapelement_t;
typedef struct hashelement_s hashelement_t;
typedef struct phnhashelement_s phnhashelement_t;
typedef struct dicthashelement_s dicthashelement_t;

/* External functions from mk_mdef_gen.c */
extern int32 make_ci_list_cd_hash_frm_phnlist(const char *phnlist,
                                              char ***CIlist,
                                              int32 *cilistsize,
                                              hashelement_t ***CDhash,
                                              int32 *NCDphones);

extern int32 make_ci_list_frm_mdef(const char *mdeffile,
                                   char ***CIlist,
                                   int32 *cilistsize);

extern int32 read_dict(const char *dictfile, const char *fillerdictfile,
                       dicthashelement_t ***dicthash);

extern int32 make_dict_triphone_list(dicthashelement_t **dicthash,
                                     hashelement_t ***triphonehash,
                                     int ignore_wpos);

extern int32 count_triphones(const char *transfile,
                             dicthashelement_t **dicthash,
                             hashelement_t **tphnhash,
                             phnhashelement_t ***CIhash,
                             int ignore_wpos);

extern int32 find_threshold(hashelement_t **triphonehash);

extern int32 make_CD_heap(hashelement_t **triphonehash,
                          int32 threshold,
                          heapelement_t ***CDheap,
                          int32 *cdheapsize);

extern int32 make_mdef_from_list(const char *mdeffile,
                                 char **CIlist,
                                 int32 cilistsize,
                                 heapelement_t **CDheap,
                                 int32 cdheapsize,
                                 char *pgm);

extern int32 print_counts(const char *countfn, phnhashelement_t **CIhash,
                          hashelement_t **CDhash);

extern void freehash(hashelement_t **hash);


/* Helper to initialize cmd_ln with required args */
static cmd_ln_t *
init_cmd_ln(uint32 n_state)
{
    char n_state_str[16];
    snprintf(n_state_str, sizeof(n_state_str), "%u", n_state);

    static const arg_t args[] = {
        { "-n_state_pm", ARG_INT32, "3", "States per phone" },
        { "-minocc", ARG_INT32, "1", "Min occurrences" },
        { "-maxtriphones", ARG_INT32, "100000", "Max triphones" },
        { "-ignorewpos", ARG_BOOLEAN, "no", "Ignore word position" },
        { NULL, 0, NULL, NULL }
    };

    /* Use global cmd_ln since mk_mdef_gen uses it */
    const char *argv[] = { "pstrain_mdef_gen", "-n_state_pm", n_state_str };
    cmd_ln_parse(args, 3, (char **)argv, FALSE);

    return NULL;  /* Using global cmd_ln */
}


int
pstrain_mdef_gen_ci(const char *phone_list_path,
                const char *output_path,
                uint32 n_state)
{
    char **CIlist = NULL;
    hashelement_t **CDhash = NULL;
    int32 cilistsize = 0;
    int32 ncd = 0;
    int ret = -1;

    if (!phone_list_path || !output_path) {
        E_ERROR("NULL path argument\n");
        return -1;
    }

    init_cmd_ln(n_state);

    /* Parse phone list to get CI phones */
    if (make_ci_list_cd_hash_frm_phnlist(phone_list_path, &CIlist,
                                         &cilistsize, &CDhash, &ncd) != S3_SUCCESS) {
        E_ERROR("Failed to parse phone list %s\n", phone_list_path);
        goto cleanup;
    }

    /* Generate CI-only mdef (NULL CDheap, 0 cdheapsize) */
    if (make_mdef_from_list(output_path, CIlist, cilistsize,
                            NULL, 0, "pstrain_mdef_gen") != S3_SUCCESS) {
        E_ERROR("Failed to write CI mdef to %s\n", output_path);
        goto cleanup;
    }

    E_INFO("Wrote CI mdef with %d phones to %s\n", cilistsize, output_path);
    ret = 0;

cleanup:
    if (CIlist) ckd_free_2d((void **)CIlist);
    if (CDhash) freehash(CDhash);
    return ret;
}


int
pstrain_mdef_gen_alltriphones(const char *phone_list_path,
                          const char *dict_path,
                          const char *filler_dict_path,
                          const char *output_path,
                          uint32 n_state,
                          int32 ignore_wpos)
{
    char **CIlist = NULL;
    hashelement_t **CDhash = NULL;
    dicthashelement_t **dicthash = NULL;
    heapelement_t **CDheap = NULL;
    int32 cilistsize = 0;
    int32 ncd = 0;
    int32 cdheapsize = 0;
    int ret = -1;

    if (!phone_list_path || !dict_path || !output_path) {
        E_ERROR("NULL path argument\n");
        return -1;
    }

    init_cmd_ln(n_state);

    /* Set ignorewpos if requested */
    if (ignore_wpos) {
        cmd_ln_set_boolean("-ignorewpos", TRUE);
    }

    /* Parse phone list to get CI phones */
    if (make_ci_list_cd_hash_frm_phnlist(phone_list_path, &CIlist,
                                         &cilistsize, &CDhash, &ncd) != S3_SUCCESS) {
        E_ERROR("Failed to parse phone list %s\n", phone_list_path);
        goto cleanup;
    }

    /* Read dictionary (returns vocab size, 0 or negative on error) */
    if (read_dict(dict_path, filler_dict_path, &dicthash) <= 0) {
        E_ERROR("Failed to read dictionary %s\n", dict_path);
        goto cleanup;
    }

    /* Generate triphones from dictionary */
    if (CDhash) freehash(CDhash);
    CDhash = NULL;
    if (make_dict_triphone_list(dicthash, &CDhash, ignore_wpos) != S3_SUCCESS) {
        E_ERROR("Failed to generate triphone list from dictionary\n");
        goto cleanup;
    }

    /* Build heap with all triphones (threshold = -1 means all) */
    if (make_CD_heap(CDhash, -1, &CDheap, &cdheapsize) != S3_SUCCESS) {
        E_ERROR("Failed to build triphone heap\n");
        goto cleanup;
    }

    /* Generate mdef with all triphones */
    if (make_mdef_from_list(output_path, CIlist, cilistsize,
                            CDheap, cdheapsize, "pstrain_mdef_gen") != S3_SUCCESS) {
        E_ERROR("Failed to write all-triphones mdef to %s\n", output_path);
        goto cleanup;
    }

    E_INFO("Wrote all-triphones mdef with %d CI + %d CD phones to %s\n",
           cilistsize, cdheapsize, output_path);
    ret = 0;

cleanup:
    if (CIlist) ckd_free_2d((void **)CIlist);
    if (CDhash) freehash(CDhash);
    /* Note: CDheap elements are freed inside make_mdef_from_list */
    return ret;
}


int
pstrain_mdef_gen_untied(const char *phone_list_path,
                    const char *dict_path,
                    const char *filler_dict_path,
                    const char *transcript_path,
                    const char *output_path,
                    uint32 n_state,
                    int32 ignore_wpos,
                    int32 multipron)
{
    char **CIlist = NULL;
    hashelement_t **CDhash = NULL;
    dicthashelement_t **dicthash = NULL;
    phnhashelement_t **CIhash = NULL;
    heapelement_t **CDheap = NULL;
    int32 cilistsize = 0;
    int32 ncd = 0;
    int32 cdheapsize = 0;
    int ret = -1;

    if (!phone_list_path || !dict_path || !transcript_path || !output_path) {
        E_ERROR("NULL path argument\n");
        return -1;
    }

    init_cmd_ln(n_state);

    if (ignore_wpos) {
        cmd_ln_set_boolean("-ignorewpos", TRUE);
    }

    /* Parse phone list to get CI phones */
    if (make_ci_list_cd_hash_frm_phnlist(phone_list_path, &CIlist,
                                         &cilistsize, &CDhash, &ncd) != S3_SUCCESS) {
        E_ERROR("Failed to parse phone list %s\n", phone_list_path);
        goto cleanup;
    }

    /* Read dictionary (returns vocab size, 0 or negative on error) */
    if (read_dict(dict_path, filler_dict_path, &dicthash) <= 0) {
        E_ERROR("Failed to read dictionary %s\n", dict_path);
        goto cleanup;
    }

    /* Generate triphones from dictionary */
    if (CDhash) freehash(CDhash);
    CDhash = NULL;
    if (make_dict_triphone_list(dicthash, &CDhash, ignore_wpos) != S3_SUCCESS) {
        E_ERROR("Failed to generate triphone list from dictionary\n");
        goto cleanup;
    }

    /* Count triphones in transcripts */
    if (count_triphones(transcript_path, dicthash, CDhash, &CIhash, ignore_wpos) != S3_SUCCESS) {
        E_ERROR("Failed to count triphones in %s\n", transcript_path);
        goto cleanup;
    }

    /* In multipron mode the BW pronunciation graph can select every dictionary variant and
     * constructs cross-word left/right context products.  count_triphones()
     * follows only the primary dictionary entry, so occurrence pruning here
     * omits models that the graph is allowed to request.  Such slots fall
     * back to CI states; after those zero-occupancy states are re-estimated,
     * an otherwise valid utterance path can become numerically dead.  Keep
     * the complete dictionary-producible set already enumerated above so
     * graph topology and untied-model initialization have the same domain.
     * The linear path retains upstream mk_mdef_gen semantics: only triphones
     * with a positive transcript occurrence are emitted. */
    if (make_CD_heap(CDhash, multipron ? -1 : find_threshold(CDhash),
                     &CDheap, &cdheapsize) != S3_SUCCESS) {
        E_ERROR("Failed to build triphone heap\n");
        goto cleanup;
    }

    /* Generate untied mdef */
    if (make_mdef_from_list(output_path, CIlist, cilistsize,
                            CDheap, cdheapsize, "pstrain_mdef_gen") != S3_SUCCESS) {
        E_ERROR("Failed to write untied mdef to %s\n", output_path);
        goto cleanup;
    }

    E_INFO("Wrote untied mdef with %d CI + %d CD phones to %s\n",
           cilistsize, cdheapsize, output_path);
    ret = 0;

cleanup:
    if (CIlist) ckd_free_2d((void **)CIlist);
    if (CDhash) freehash(CDhash);
    /* CIhash cleanup would need freephnhash - TODO */
    return ret;
}


int
pstrain_mdef_count_triphones(const char *phone_list_path,
                         const char *dict_path,
                         const char *filler_dict_path,
                         const char *transcript_path,
                         const char *output_path,
                         int32 ignore_wpos)
{
    char **CIlist = NULL;
    hashelement_t **CDhash = NULL;
    dicthashelement_t **dicthash = NULL;
    phnhashelement_t **CIhash = NULL;
    int32 cilistsize = 0;
    int32 ncd = 0;
    int ret = -1;

    if (!phone_list_path || !dict_path || !transcript_path || !output_path) {
        E_ERROR("NULL path argument\n");
        return -1;
    }

    init_cmd_ln(3);  /* n_state doesn't matter for counting */

    if (ignore_wpos) {
        cmd_ln_set_boolean("-ignorewpos", TRUE);
    }

    /* Parse phone list */
    if (make_ci_list_cd_hash_frm_phnlist(phone_list_path, &CIlist,
                                         &cilistsize, &CDhash, &ncd) != S3_SUCCESS) {
        E_ERROR("Failed to parse phone list %s\n", phone_list_path);
        goto cleanup;
    }

    /* Read dictionary (returns vocab size, 0 or negative on error) */
    if (read_dict(dict_path, filler_dict_path, &dicthash) <= 0) {
        E_ERROR("Failed to read dictionary %s\n", dict_path);
        goto cleanup;
    }

    /* Generate triphones from dictionary */
    if (CDhash) freehash(CDhash);
    CDhash = NULL;
    if (make_dict_triphone_list(dicthash, &CDhash, ignore_wpos) != S3_SUCCESS) {
        E_ERROR("Failed to generate triphone list\n");
        goto cleanup;
    }

    /* Count triphones */
    if (count_triphones(transcript_path, dicthash, CDhash, &CIhash, ignore_wpos) != S3_SUCCESS) {
        E_ERROR("Failed to count triphones\n");
        goto cleanup;
    }

    /* Print counts */
    if (print_counts(output_path, CIhash, CDhash) != S3_SUCCESS) {
        E_ERROR("Failed to write counts to %s\n", output_path);
        goto cleanup;
    }

    E_INFO("Wrote triphone counts to %s\n", output_path);
    ret = 0;

cleanup:
    if (CIlist) ckd_free_2d((void **)CIlist);
    if (CDhash) freehash(CDhash);
    return ret;
}
