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
 * @file pstrain_dtree.c
 * @brief CFFI-friendly wrappers for decision tree operations.
 */

#include "pstrain_dtree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>

#include <s3/model_def_io.h>
#include <s3/s3mixw_io.h>
#include <s3/s3gau_io.h>
#include <s3/s3tmat_io.h>
#include <s3/s3ts2cb_io.h>
#include <s3/ts2cb.h>
#include <s3/gauden.h>
#include <s3/s3.h>
#include <s3/dtree.h>
#include <s3/quest.h>
#include <s3/pset_io.h>
#include <s3/acmod_set.h>
#include <s3/was_added.h>

static uint64
mixw_digest(const float32 *data, size_t n_float)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t i;
    uint64 digest = 1469598103934665603ULL;

    for (i = 0; i < n_float * sizeof(*data); i++) {
        digest ^= bytes[i];
        digest *= 1099511628211ULL;
    }
    return digest;
}


pset_t *
pstrain_read_pset(const char *filename,
              const char *mdef_path,
              uint32 *out_n_pset)
{
    model_def_t *mdef = NULL;
    pset_t *pset;
    uint32 n_pset;

    if (!filename || !mdef_path || !out_n_pset) {
        E_ERROR("NULL argument\n");
        return NULL;
    }

    if (model_def_read(&mdef, mdef_path) != S3_SUCCESS) {
        E_ERROR("Failed to read mdef: %s\n", mdef_path);
        return NULL;
    }

    pset = read_pset_file(filename, mdef->acmod_set, &n_pset);
    if (pset == NULL) {
        E_ERROR("Failed to read pset: %s\n", filename);
        return NULL;
    }

    *out_n_pset = n_pset;

    /* Note: mdef is leaked here. For a proper implementation,
     * we'd need to track it for later cleanup. */

    return pset;
}


void
pstrain_free_pset(pset_t *pset, uint32 n_pset)
{
    uint32 i;
    if (pset) {
        for (i = 0; i < n_pset; i++) {
            if (pset[i].name) ckd_free(pset[i].name);
            if (pset[i].phone) ckd_free(pset[i].phone);
            if (pset[i].member) ckd_free(pset[i].member);
            if (pset[i].posn) ckd_free(pset[i].posn);
        }
        ckd_free(pset);
    }
}


/* Helper: find triphone range for a base phone */
static int
find_triphones(model_def_t *mdef, const char *phn, uint32 *p_s, uint32 *p_e)
{
    uint32 targ_base, p, b;

    targ_base = acmod_set_name2id(mdef->acmod_set, phn);
    *p_s = *p_e = NO_ID;

    if (targ_base == NO_ACMOD)
        return -1;
    if (targ_base >= acmod_set_n_ci(mdef->acmod_set)) {
        *p_s = *p_e = targ_base;
        return 0;
    }

    for (p = acmod_set_n_ci(mdef->acmod_set);
         p < acmod_set_n_acmod(mdef->acmod_set); p++) {
        b = acmod_set_base_phone(mdef->acmod_set, p);
        if ((b == targ_base) && (*p_s == NO_ID)) {
            *p_s = p;
        }
        else if ((b == targ_base) && (*p_s != NO_ID) && (*p_e != NO_ID)) {
            E_FATAL("n-phones with base phone %s occur non-consecutively\n", phn);
        }
        else if ((b != targ_base) && (*p_s != NO_ID) && (*p_e == NO_ID)) {
            *p_e = p-1;
        }
    }

    if ((p == acmod_set_n_acmod(mdef->acmod_set)) && (*p_e == NO_ID)) {
        *p_e = p-1;
    }

    if (*p_s == NO_ID) {
        E_WARN("No triphones involving %s\n", phn);
        return -1;
    }
    return 0;
}


int
pstrain_build_tree(const char *mdef_path,
               const char *mixw_path,
               const char *mean_path,
               const char *var_path,
               const char *pset_path,
               const char *output_path,
               const char *phone,
               uint32 state,
               int32 continuous,
               uint32 ssplitmin,
               uint32 ssplitmax,
               float32 ssplitthr,
               uint32 csplitmin,
               uint32 csplitmax,
               float32 csplitthr,
               float32 mwfloor,
               float32 varfloor,
               float32 cntthresh,
               float32 *stwt,
               uint32 n_stwt,
               int32 rotate_state_weights,
               int32 directional_questions,
               int32 allphones,
               int32 intermediate_dumps)
{
    model_def_t *mdef = NULL;
    pset_t *pset = NULL;
    uint32 n_pset;
    uint32 p_s, p_e;
    uint32 n_model, n_state, n_stream, n_density;
    uint32 mixw_s, mixw_e;
    float32 ***in_mixw = NULL;
    float32 *in_mixw_snapshot = NULL;
    size_t in_mixw_n_float;
    uint64 in_mixw_entry_digest;
    float32 ****mixw = NULL;
    float32 ****mixw_occ = NULL;
    float32 ****means = NULL;
    float32 ****vars = NULL;
    uint32 *veclen = NULL;
    uint32 **dfeat = NULL;
    quest_t *all_q = NULL;
    uint32 n_all_q;
    uint32 *id = NULL;
    uint32 *model_acmod = NULL;
    float32 *state_wt = NULL;
    dtree_t *tr;
    FILE *fp;
    uint32 i, j, k, m, s;
    uint32 n_in_mixw;
    float64 norm, dnom;
    acmod_id_t b, l, r;
    word_posn_t pn;

    char varfloor_str[32];

    /* Validate inputs */
    if (!mdef_path || !mixw_path || !pset_path || !output_path) {
        E_ERROR("NULL required path argument\n");
        return -1;
    }
    if (!allphones && !phone) {
        E_ERROR("Must specify phone or allphones\n");
        return -1;
    }
    if (continuous && (!mean_path || !var_path)) {
        E_ERROR("Continuous mode requires mean and var paths\n");
        return -1;
    }

    /* Set up command line arguments that internal functions expect */
    snprintf(varfloor_str, sizeof(varfloor_str), "%g", varfloor);
    {
        static const arg_t args[] = {
            { "-ts2cbfn", ARG_STRING, ".semi.", "HMM type (.cont. or .semi.)" },
            { "-varfloor", ARG_FLOAT32, "1e-5", "Variance floor" },
            { NULL, 0, NULL, NULL }
        };
        const char *type_str = continuous ? ".cont." : ".semi.";
        int argc = 0;
        const char *argv[8];
        argv[argc++] = "pstrain_build_tree";
        argv[argc++] = "-ts2cbfn";
        argv[argc++] = type_str;
        argv[argc++] = "-varfloor";
        argv[argc++] = varfloor_str;
        cmd_ln_parse(args, argc, (char **)argv, TRUE);
    }

    /* Read model definition */
    E_INFO("Reading mdef: %s\n", mdef_path);
    if (model_def_read(&mdef, mdef_path) != S3_SUCCESS) {
        E_ERROR("Failed to read mdef\n");
        return -1;
    }

    /* Find triphone range */
    if (allphones) {
        p_s = acmod_set_n_ci(mdef->acmod_set);
        p_e = acmod_set_n_acmod(mdef->acmod_set) - 1;
    } else {
        if (find_triphones(mdef, phone, &p_s, &p_e) != 0) {
            E_ERROR("Could not find triphones for %s\n", phone);
            return -1;
        }
    }

    E_INFO("Building tree for phones %u through %u\n", p_s, p_e);

    n_state = mdef->defn[p_s].n_state - 1;
    n_model = p_e - p_s + 1;
    for (i = p_s + 1; i <= p_e; i++) {
        uint32 model_n_state = mdef->defn[i].n_state - 1;
        if (model_n_state != n_state) {
            E_FATAL("Model topology mismatch for model %u: expected %u emitting states, got %u\n",
                    i, n_state, model_n_state);
        }
    }

    /* Find mixw range */
    mixw_s = mdef->defn[p_s].state[0];
    mixw_e = mdef->defn[p_e].state[mdef->defn[p_e].n_state-2];

    /* Read mixture weights */
    E_INFO("Reading mixw: %s\n", mixw_path);
    if (s3mixw_intv_read(mixw_path, mixw_s, mixw_e,
                         &in_mixw, &n_in_mixw, &n_stream, &n_density) != S3_SUCCESS) {
        E_ERROR("Failed to read mixw\n");
        return -1;
    }
    in_mixw_n_float = (size_t)n_in_mixw * n_stream * n_density;
    in_mixw_snapshot = ckd_calloc(in_mixw_n_float, sizeof(*in_mixw_snapshot));
    memcpy(in_mixw_snapshot, in_mixw[0][0],
           in_mixw_n_float * sizeof(*in_mixw_snapshot));
    in_mixw_entry_digest = mixw_digest(in_mixw_snapshot, in_mixw_n_float);

    /* Allocate mixw arrays */
    /* Keep occupancies independently owned.  The continuous-model collapse
     * below replaces density zero with the stream total; pointing these rows
     * into in_mixw would silently mutate the parsed, read-only input and make
     * later consumers observe order-dependent values. */
    mixw_occ = (float32 ****)ckd_calloc_4d(n_model, n_state, n_stream,
                                           n_density, sizeof(float32));
    mixw = (float32 ****)ckd_calloc_4d(n_model, n_state, n_stream, n_density, sizeof(float32));

    model_acmod = ckd_calloc(n_model, sizeof(uint32));

    /* Match upstream bldtree/main.c: a model participates in question
     * statistics only when every emitting-state/stream occupancy is at
     * least -cntthresh (default 0.00001).  This matters for dictionary-wide
     * untied inventories, whose unobserved models have zero occupancy. */
    for (i = p_s, j = 0; i <= p_e; i++) {
        int include = TRUE;
        for (k = 0; k < n_state; k++) {
            s = mdef->defn[i].state[k] - mixw_s;
            for (m = 0; m < n_stream; m++) {
                float32 count = 0.0f;
                uint32 density;
                for (density = 0; density < n_density; density++)
                    count += in_mixw[s][m][density];
                if (count < cntthresh)
                    include = FALSE;
            }
        }
        if (!include)
            continue;
        model_acmod[j] = i;
        for (k = 0; k < n_state; k++) {
            s = mdef->defn[i].state[k] - mixw_s;
            for (m = 0; m < n_stream; m++)
                memcpy(mixw_occ[j][k][m], in_mixw[s][m],
                       n_density * sizeof(float32));
            if (mixw_occ[j][k][0] == in_mixw[s][0])
                E_FATAL("mixw_occ must not alias parsed mixture weights\n");
        }
        j++;
    }

    /* This stage is declared read-only with respect to parsed input.  Keep
     * both byte identity and an independent entry/exit digest as regression
     * guards: either changing means a write escaped through an alias. */
    if (memcmp(in_mixw_snapshot, in_mixw[0][0],
               in_mixw_n_float * sizeof(*in_mixw_snapshot)) != 0
        || mixw_digest(in_mixw[0][0], in_mixw_n_float) != in_mixw_entry_digest)
        E_FATAL("model selection mutated parsed mixture weights\n");
    E_INFO("%u of %u models have observation count at least %f\n",
           j, p_e - p_s + 1, cntthresh);
    n_model = j;
    if (n_model == 0) {
        E_ERROR("No models meet count threshold %f\n", cntthresh);
        return -1;
    }

    /* Normalize mixture weights */
    for (s = 0; s < n_state; s++) {
        for (i = 0; i < n_model; i++) {
            for (j = 0; j < n_stream; j++) {
                for (k = 0, dnom = 0; k < n_density; k++) {
                    dnom += mixw_occ[i][s][j][k];
                }
                if (dnom != 0) {
                    norm = 1.0 / dnom;
                    for (k = 0; k < n_density; k++) {
                        mixw[i][s][j][k] = mixw_occ[i][s][j][k] * norm;
                        if (mixw[i][s][j][k] < mwfloor)
                            mixw[i][s][j][k] = mwfloor;
                    }
                }
            }
        }
    }
    if (memcmp(in_mixw_snapshot, in_mixw[0][0],
               in_mixw_n_float * sizeof(*in_mixw_snapshot)) != 0
        || mixw_digest(in_mixw[0][0], in_mixw_n_float) != in_mixw_entry_digest)
        E_FATAL("normalization mutated parsed mixture weights\n");

    /* Read means and variances for continuous models */
    if (continuous) {
        vector_t ***fullmean = NULL, ***fullvar = NULL;
        uint32 l_nstates, t_nstates, t_nfeat, t_ndensity;
        uint32 *l_veclen = NULL, *t_veclen = NULL;
        uint32 sumveclen, ll, n, nn;

        E_INFO("Reading means: %s\n", mean_path);
        if (s3gau_read(mean_path, &fullmean, &l_nstates, &t_nfeat, &t_ndensity, &l_veclen) != S3_SUCCESS) {
            E_ERROR("Failed to read means\n");
            return -1;
        }
        veclen = l_veclen;
        if (t_nfeat != n_stream && t_ndensity != n_density) {
            E_FATAL("Mean/mixture-weight dimension mismatch: expected %u streams and %u densities, got %u streams and %u densities\n",
                    n_stream, n_density, t_nfeat, t_ndensity);
        }

        E_INFO("Reading vars: %s\n", var_path);
        if (s3gau_read(var_path, &fullvar, &t_nstates, &t_nfeat, &t_ndensity, &t_veclen) != S3_SUCCESS) {
            E_ERROR("Failed to read vars\n");
            return -1;
        }
        if (t_nfeat != n_stream && t_ndensity != n_density) {
            E_FATAL("Variance/mixture-weight dimension mismatch: expected %u streams and %u densities, got %u streams and %u densities\n",
                    n_stream, n_density, t_nfeat, t_ndensity);
        }
        for (i = 0; i < n_stream; i++) {
            if (t_veclen[i] != l_veclen[i]) {
                E_FATAL("Mean/variance vector-length mismatch for stream %u: expected %u, got %u\n",
                        i, l_veclen[i], t_veclen[i]);
            }
        }
        if (t_nstates != l_nstates) {
            E_FATAL("Mean/variance state-count mismatch: expected %u, got %u\n",
                    l_nstates, t_nstates);
        }

        /* Allocate and compute merged Gaussians */
        for (i = 0, sumveclen = 0; i < n_stream; i++)
            sumveclen += l_veclen[i];

        means = (float32 ****)ckd_calloc_4d(n_model, n_state, n_stream, sumveclen, sizeof(float32));
        vars = (float32 ****)ckd_calloc_4d(n_model, n_state, n_stream, sumveclen, sizeof(float32));

        for (j = 0; j < n_model; j++) {
            i = model_acmod[j];
            for (k = 0; k < n_state; k++) {
                /* Use actual tied state index from mdef, not consecutive m++ */
                uint32 ts = mdef->defn[i].state[k];
                for (ll = 0; ll < n_stream; ll++) {
                    float32 *featmean = means[j][k][ll];
                    float32 *featvar = vars[j][k][ll];
                    dnom = 0;
                    for (n = 0; n < t_ndensity; n++) {
                        float32 mw = mixw_occ[j][k][ll][n];
                        dnom += mw;
                        for (nn = 0; nn < l_veclen[ll]; nn++) {
                            featmean[nn] += mw * fullmean[ts][ll][n][nn];
                            featvar[nn] += mw * (fullmean[ts][ll][n][nn] * fullmean[ts][ll][n][nn] +
                                                 fullvar[ts][ll][n][nn]);
                        }
                    }
                    if (dnom != 0) {
                        for (nn = 0; nn < l_veclen[ll]; nn++) {
                            featmean[nn] /= dnom;
                            featvar[nn] = featvar[nn] / dnom - featmean[nn] * featmean[nn];
                            if (featvar[nn] < varfloor)
                                featvar[nn] = varfloor;
                        }
                    }
                    mixw_occ[j][k][ll][0] = dnom;
                }
            }
        }

        /* For continuous, use 1 density */
        n_density = 1;

        ckd_free_4d((void ****)fullmean);
        ckd_free_4d((void ****)fullvar);
        ckd_free(t_veclen);
    }

    if (memcmp(in_mixw_snapshot, in_mixw[0][0],
               in_mixw_n_float * sizeof(*in_mixw_snapshot)) != 0
        || mixw_digest(in_mixw[0][0], in_mixw_n_float) != in_mixw_entry_digest)
        E_FATAL("continuous-model collapse mutated parsed mixture weights\n");

    /* Read phone sets */
    E_INFO("Reading pset: %s\n", pset_path);
    pset = read_pset_file(pset_path, mdef->acmod_set, &n_pset);
    if (pset == NULL) {
        E_ERROR("Failed to read pset\n");
        return -1;
    }

    /* Build decision tree features */
    dfeat = (uint32 **)ckd_calloc_2d(n_model, 4, sizeof(uint32));
    for (j = 0; j < n_model; j++) {
        i = model_acmod[j];
        acmod_set_id2tri(mdef->acmod_set, &b, &l, &r, &pn, i);
        dfeat[j][0] = (uint32)l;
        dfeat[j][1] = (uint32)b;
        dfeat[j][2] = (uint32)r;
        dfeat[j][3] = (uint32)pn;
    }

    /* Generate simple questions from phone sets */
    {
        uint32 n_phone_q = 0, n_wdbndry = 0;
        uint32 qi;

        for (i = 0; i < n_pset; i++) {
            if (pset[i].member) {
                uint32 phone_id;
                int has_member = FALSE;
                for (phone_id = 0; phone_id < acmod_set_n_ci(mdef->acmod_set); phone_id++) {
                    if (pset[i].member[phone_id]) {
                        has_member = TRUE;
                        break;
                    }
                }
                if (!has_member) {
                    E_FATAL("Invalid phone-set record %u (%s): expected at least one known phone member or a word-position marker\n",
                            i, pset[i].name ? pset[i].name : "<unnamed>");
                }
                if (directional_questions
                    && (strstr(pset[i].name, "_L") != NULL
                        || strstr(pset[i].name, "_R") != NULL))
                    n_phone_q += 1;
                else
                    n_phone_q += 2;  /* left and right context */
            }
            else if (pset[i].posn)
                n_wdbndry++;
            else {
                E_FATAL("Invalid phone-set record %u (%s): expected members or a word-position marker\n",
                        i, pset[i].name ? pset[i].name : "<unnamed>");
            }
        }
        n_all_q = 2 * n_phone_q + 2 * n_wdbndry;

        all_q = ckd_calloc(n_all_q, sizeof(quest_t));

        for (i = 0, qi = 0; i < n_pset; i++) {
            if (pset[i].member) {
                int left = !directional_questions
                    || strstr(pset[i].name, "_L") != NULL
                    || (strstr(pset[i].name, "_L") == NULL
                        && strstr(pset[i].name, "_R") == NULL);
                int right = !directional_questions
                    || strstr(pset[i].name, "_R") != NULL
                    || (strstr(pset[i].name, "_L") == NULL
                        && strstr(pset[i].name, "_R") == NULL);
                int neg;

                for (neg = FALSE; neg <= TRUE; neg++) {
                    if (left) {
                        all_q[qi].pset = i;
                        all_q[qi].member = pset[i].member;
                        all_q[qi].neg = neg;
                        all_q[qi].ctxt = -1;
                        qi++;
                    }
                    if (right) {
                        all_q[qi].pset = i;
                        all_q[qi].member = pset[i].member;
                        all_q[qi].neg = neg;
                        all_q[qi].ctxt = 1;
                        qi++;
                    }
                }
            }
            else if (pset[i].posn) {
                all_q[qi].pset = i;
                all_q[qi].posn = pset[i].posn;
                all_q[qi].neg = FALSE;
                qi++;

                all_q[qi].pset = i;
                all_q[qi].posn = pset[i].posn;
                all_q[qi].neg = TRUE;
                qi++;
            }
        }
        n_all_q = qi;
    }

    /* Set up state weights */
    state_wt = ckd_calloc(n_state, sizeof(float32));
    if (stwt && n_stwt != n_state) {
        E_ERROR("State weight count mismatch: expected %u, got %u\n", n_state, n_stwt);
        return -(int)n_state - 2;
    }
    if (stwt) {
        for (i = 0; i < n_state; i++) {
            uint32 distance = i > state ? i - state : state - i;
            state_wt[i] = stwt[rotate_state_weights ? distance : i];
        }
    } else {
        /* Uniform weights */
        for (i = 0; i < n_state; i++)
            state_wt[i] = 1.0 / n_state;
    }

    /* Normalize weights */
    for (i = 0, norm = 0; i < n_state; i++)
        norm += state_wt[i];
    norm = 1.0 / norm;
    for (i = 0; i < n_state; i++)
        state_wt[i] *= norm;

    /* Initialize id array */
    id = ckd_calloc(n_model, sizeof(uint32));
    for (i = 0; i < n_model; i++)
        id[i] = i;

    /* Build the tree */
    E_INFO("Building composite tree...\n");

    tr = mk_tree_comp(mixw_occ, means, vars, veclen, n_model, n_state,
                      n_stream, n_density, state_wt,
                      id, n_model,
                      all_q, n_all_q, pset, acmod_set_n_ci(mdef->acmod_set),
                      dfeat, 4,
                      ssplitmin, ssplitmax, ssplitthr,
                      csplitmin, csplitmax, csplitthr,
                      mwfloor, intermediate_dumps);

    if (memcmp(in_mixw_snapshot, in_mixw[0][0],
               in_mixw_n_float * sizeof(*in_mixw_snapshot)) != 0
        || mixw_digest(in_mixw[0][0], in_mixw_n_float) != in_mixw_entry_digest)
        E_FATAL("tree construction mutated parsed mixture weights\n");

    if (tr == NULL) {
        E_ERROR("Failed to build tree\n");
        return -1;
    }

    /* Write the tree */
    E_INFO("Writing tree to: %s\n", output_path);
    fp = fopen(output_path, "w");
    if (fp == NULL) {
        E_ERROR("Unable to open %s for writing\n", output_path);
        return -1;
    }
    print_final_tree(fp, &tr->node[0], pset);
    fclose(fp);

    /* Cleanup */
    free_tree(tr);
    ckd_free(id);
    ckd_free(model_acmod);
    ckd_free(state_wt);
    ckd_free(all_q);
    ckd_free_2d((void **)dfeat);
    ckd_free_4d((void ****)mixw);
    ckd_free_4d((void ****)mixw_occ);
    ckd_free_3d((void ***)in_mixw);
    ckd_free(in_mixw_snapshot);
    if (means) ckd_free_4d((void ****)means);
    if (vars) ckd_free_4d((void ****)vars);

    E_INFO("Tree built successfully\n");
    return 0;
}


/* External function from tiestate/main.c */
extern int tiestate_run(void);

int
pstrain_tie_states(const char *input_mdef_path,
               const char *output_mdef_path,
               const char *tree_dir,
               const char *pset_path,
               const char *phone,
               int32 allphones)
{
    int ret;

    if (!input_mdef_path || !output_mdef_path || !tree_dir || !pset_path) {
        E_ERROR("NULL required path argument\n");
        return -1;
    }

    static const arg_t args[] = {
        { "-imoddeffn", ARG_STRING, NULL, "Input model definition file" },
        { "-omoddeffn", ARG_STRING, NULL, "Output model definition file" },
        { "-treedir", ARG_STRING, NULL, "Directory with decision trees" },
        { "-psetfn", ARG_STRING, NULL, "Phone set file" },
        { "-allphones", ARG_BOOLEAN, "no", "All phones" },
        { "-help", ARG_BOOLEAN, "no", "Help" },
        { NULL, 0, NULL, NULL }
    };

    int argc = 0;
    const char *argv[16];
    argv[argc++] = "pstrain_tie_states";
    argv[argc++] = "-imoddeffn";
    argv[argc++] = input_mdef_path;
    argv[argc++] = "-omoddeffn";
    argv[argc++] = output_mdef_path;
    argv[argc++] = "-treedir";
    argv[argc++] = tree_dir;
    argv[argc++] = "-psetfn";
    argv[argc++] = pset_path;
    if (allphones) {
        argv[argc++] = "-allphones";
        argv[argc++] = "yes";
    }

    cmd_ln_parse(args, argc, (char **)argv, TRUE);

    ret = tiestate_run();

    if (ret != 0) {
        E_ERROR("tiestate_run failed\n");
        return -1;
    }

    E_INFO("Wrote tied mdef to %s\n", output_mdef_path);
    return 0;
}


/* External function from make_quests/main.c */
extern int make_quests_run(void);
/* External function from prunetree/main.c */
extern int prunetree_run(void);

int
pstrain_make_quests(const char *mdef_path,
                const char *mixw_path,
                const char *mean_path,
                const char *var_path,
                const char *output_path,
                int32 continuous,
                uint32 npermute,
                uint32 quests_per_state,
                float32 varfloor,
                uint32 niter)
{
    char npermute_str[16], qstperstt_str[16], varfloor_str[32], niter_str[16];
    int ret;

    if (!mdef_path || !mixw_path || !output_path) {
        E_ERROR("NULL required path argument\n");
        return -1;
    }

    if (continuous && (!mean_path || !var_path)) {
        E_ERROR("Continuous mode requires mean_path and var_path\n");
        return -1;
    }

    /* Set up command line arguments */
    snprintf(npermute_str, sizeof(npermute_str), "%u", npermute);
    snprintf(qstperstt_str, sizeof(qstperstt_str), "%u", quests_per_state);
    snprintf(varfloor_str, sizeof(varfloor_str), "%g", varfloor);
    snprintf(niter_str, sizeof(niter_str), "%u", niter);

    static const arg_t args[] = {
        { "-moddeffn", ARG_STRING, NULL, "Model definition file" },
        { "-mixwfn", ARG_STRING, NULL, "Mixture weights file" },
        { "-meanfn", ARG_STRING, NULL, "Means file" },
        { "-varfn", ARG_STRING, NULL, "Variance file" },
        { "-questfn", ARG_STRING, NULL, "Output question file" },
        { "-type", ARG_STRING, NULL, "HMM type (.cont. or .semi.)" },
        { "-npermute", ARG_INT32, "6", "Permutations for clustering" },
        { "-qstperstt", ARG_INT32, "8", "Questions per state" },
        { "-varfloor", ARG_FLOAT32, "1e-8", "Variance floor" },
        { "-niter", ARG_INT32, "0", "Number of iterations" },
        { "-fullvar", ARG_BOOLEAN, "no", "Full covariance" },
        { "-help", ARG_BOOLEAN, "no", "Help" },
        { "-example", ARG_BOOLEAN, "no", "Example" },
        { NULL, 0, NULL, NULL }
    };

    const char *type_str = continuous ? ".cont." : ".semi.";

    int argc = 0;
    const char *argv[32];
    argv[argc++] = "pstrain_make_quests";
    argv[argc++] = "-moddeffn";
    argv[argc++] = mdef_path;
    argv[argc++] = "-mixwfn";
    argv[argc++] = mixw_path;
    argv[argc++] = "-questfn";
    argv[argc++] = output_path;
    argv[argc++] = "-type";
    argv[argc++] = type_str;
    argv[argc++] = "-npermute";
    argv[argc++] = npermute_str;
    argv[argc++] = "-qstperstt";
    argv[argc++] = qstperstt_str;
    argv[argc++] = "-varfloor";
    argv[argc++] = varfloor_str;
    argv[argc++] = "-niter";
    argv[argc++] = niter_str;

    if (continuous) {
        argv[argc++] = "-meanfn";
        argv[argc++] = mean_path;
        argv[argc++] = "-varfn";
        argv[argc++] = var_path;
    }

    cmd_ln_parse(args, argc, (char **)argv, TRUE);

    ret = make_quests_run();

    if (ret != 0) {
        E_ERROR("make_quests_run failed\n");
        return -1;
    }

    E_INFO("Wrote phonetic questions to %s\n", output_path);
    return 0;
}


int
pstrain_prune_tree(const char *mdef_path,
               const char *pset_path,
               const char *input_tree_dir,
               const char *output_tree_dir,
               uint32 n_seno_target,
               float32 min_occ,
               int32 allphones)
{
    char nseno_str[16], minocc_str[32];
    int ret;

    if (!mdef_path || !pset_path || !input_tree_dir || !output_tree_dir) {
        E_ERROR("NULL required path argument\n");
        return -1;
    }

    /* Set up string arguments */
    snprintf(nseno_str, sizeof(nseno_str), "%u", n_seno_target);
    snprintf(minocc_str, sizeof(minocc_str), "%g", min_occ);

    static const arg_t args[] = {
        { "-moddeffn", ARG_STRING, NULL, "CI model definition file" },
        { "-psetfn", ARG_STRING, NULL, "Phone set definition file" },
        { "-itreedir", ARG_STRING, NULL, "Input tree directory" },
        { "-otreedir", ARG_STRING, NULL, "Output tree directory" },
        { "-nseno", ARG_INT32, NULL, "Target number of senones" },
        { "-minocc", ARG_FLOAT32, "0.0", "Minimum occupancy" },
        { "-allphones", ARG_BOOLEAN, "no", "All phones" },
        { "-help", ARG_BOOLEAN, "no", "Help" },
        { "-example", ARG_BOOLEAN, "no", "Example" },
        { NULL, 0, NULL, NULL }
    };

    int argc = 0;
    const char *argv[20];
    argv[argc++] = "pstrain_prune_tree";
    argv[argc++] = "-moddeffn";
    argv[argc++] = mdef_path;
    argv[argc++] = "-psetfn";
    argv[argc++] = pset_path;
    argv[argc++] = "-itreedir";
    argv[argc++] = input_tree_dir;
    argv[argc++] = "-otreedir";
    argv[argc++] = output_tree_dir;
    argv[argc++] = "-nseno";
    argv[argc++] = nseno_str;
    argv[argc++] = "-minocc";
    argv[argc++] = minocc_str;
    if (allphones) {
        argv[argc++] = "-allphones";
        argv[argc++] = "yes";
    }

    cmd_ln_parse(args, argc, (char **)argv, TRUE);

    ret = prunetree_run();

    if (ret != 0) {
        E_ERROR("prunetree_run failed\n");
        return -1;
    }

    E_INFO("Pruned trees written to %s\n", output_tree_dir);
    return 0;
}


/* ============================================================================
 * pstrain_init_mixw - Initialize CD model parameters from CI model
 * ============================================================================ */

/* Static variables for tracking what's been initialized */
static pair_t **init_mixw_dest_list = NULL;
static pair_t **init_cb_dest_list = NULL;
static pair_t **init_tmat_dest_list = NULL;

void
pstrain_dtree_session_reset(void)
{
    init_mixw_dest_list = NULL;
    init_cb_dest_list = NULL;
    init_tmat_dest_list = NULL;
}

static void
init_uniform_mixw(float32 ***dest_mixw,
                  model_def_entry_t *dest,
                  uint32 n_feat,
                  uint32 n_gau)
{
    float32 uniform = 1.0f / (float32)n_gau;
    unsigned int s, i, j;
    uint32 d_m;

    for (s = 0; s < dest->n_state; s++) {
        d_m = dest->state[s];
        for (i = 0; i < n_feat; i++) {
            for (j = 0; j < n_gau; j++) {
                if (d_m != TYING_NON_EMITTING)
                    dest_mixw[d_m][i][j] = uniform;
            }
        }
    }
}

static void
init_model_params(float32 ***dest_mixw,
                  vector_t ***dest_mean,
                  vector_t ***dest_var,
                  float32 ***dest_tmat,
                  model_def_entry_t *dest,
                  uint32 *dest_cb_map,
                  acmod_set_t *dest_acmod_set,
                  float32 ***src_mixw,
                  vector_t ***src_mean,
                  vector_t ***src_var,
                  float32 ***src_tmat,
                  model_def_entry_t *src,
                  uint32 *src_cb_map,
                  acmod_set_t *src_acmod_set,
                  uint32 n_feat,
                  uint32 n_gau,
                  uint32 n_state_pm,
                  const uint32 *veclen)
{
    unsigned int s, j, k, l;
    unsigned int s_m, s_mg;
    unsigned int d_m, d_mg;
    uint32 s_tmat, d_tmat;

    s_tmat = src->tmat;
    d_tmat = dest->tmat;

    if (!was_added(&init_tmat_dest_list[d_tmat], s_tmat)) {
        for (j = 0; j < n_state_pm-1; j++) {
            for (k = 0; k < n_state_pm; k++) {
                dest_tmat[d_tmat][j][k] += src_tmat[s_tmat][j][k];
            }
        }
    }

    for (s = 0; s < src->n_state; s++) {
        s_m = src->state[s];
        d_m = dest->state[s];

        if ((s_m == TYING_NON_EMITTING) && (d_m == TYING_NON_EMITTING))
            continue;

        if ((s_m != TYING_NON_EMITTING) && (d_m != TYING_NON_EMITTING)) {
            if (!was_added(&init_mixw_dest_list[d_m], s_m)) {
                for (j = 0; j < n_feat; j++) {
                    for (k = 0; k < n_gau; k++) {
                        dest_mixw[d_m][j][k] += src_mixw[s_m][j][k];
                    }
                }
            }

            s_mg = src_cb_map[s_m];
            d_mg = dest_cb_map[d_m];
            if (!was_added(&init_cb_dest_list[d_mg], s_mg)) {
                for (j = 0; j < n_feat; j++) {
                    for (k = 0; k < n_gau; k++) {
                        for (l = 0; l < veclen[j]; l++) {
                            dest_mean[d_mg][j][k][l] = src_mean[s_mg][j][k][l];
                            if (dest_var)
                                dest_var[d_mg][j][k][l] = src_var[s_mg][j][k][l];
                        }
                    }
                }
            }
        }
    }
}

int
pstrain_init_mixw(const char *src_mdef_path,
              const char *src_mixw_path,
              const char *src_mean_path,
              const char *src_var_path,
              const char *src_tmat_path,
              const char *dest_mdef_path,
              const char *dest_mixw_path,
              const char *dest_mean_path,
              const char *dest_var_path,
              const char *dest_tmat_path,
              int32 continuous)
{
    model_def_t *src_mdef = NULL;
    float32 ***src_mixw = NULL;
    vector_t ***src_mean = NULL;
    vector_t ***src_var = NULL;
    float32 ***src_tmat = NULL;

    model_def_t *dest_mdef = NULL;
    float32 ***dest_mixw = NULL;
    vector_t ***dest_mean = NULL;
    vector_t ***dest_var = NULL;
    float32 ***dest_tmat = NULL;

    uint32 n_mixw_src, n_mixw_dest;
    uint32 n_feat, tmp_n_feat;
    uint32 n_gau, tmp_n_gau;
    uint32 n_cb_src, n_cb_dest;
    uint32 n_state_pm;
    uint32 n_tmat_src, n_tmat_dest;
    uint32 *veclen = NULL, *tmp_veclen = NULL;
    uint32 n_ts, n_cb;

    uint32 m, dest_m, dest_m_base, src_m;
    acmod_id_t src_m_base;
    const char *dest_m_name, *dest_m_base_name;
    uint32 i;
    int ret = 0;

    /* Validate inputs */
    if (!src_mdef_path || !src_mixw_path || !src_mean_path ||
        !src_var_path || !src_tmat_path || !dest_mdef_path ||
        !dest_mixw_path || !dest_mean_path || !dest_var_path || !dest_tmat_path) {
        E_ERROR("NULL required path argument\n");
        return -1;
    }

    E_INFO("Reading source model definition: %s\n", src_mdef_path);
    if (model_def_read(&src_mdef, src_mdef_path) != S3_SUCCESS) {
        E_ERROR("Failed to read source mdef\n");
        return -1;
    }

    /* Set up ts2cb mapping for source */
    if (continuous) {
        E_INFO("Generating continuous ts2cb mapping for source\n");
        src_mdef->cb = cont_ts2cb(src_mdef->n_tied_state);
        n_ts = src_mdef->n_tied_state;
        n_cb = src_mdef->n_tied_state;
    } else {
        E_INFO("Generating semi-continuous ts2cb mapping for source\n");
        src_mdef->cb = semi_ts2cb(src_mdef->n_tied_state);
        n_ts = src_mdef->n_tied_state;
        n_cb = 1;
    }

    E_INFO("Reading source mixture weights: %s\n", src_mixw_path);
    if (s3mixw_read(src_mixw_path, &src_mixw, &n_mixw_src, &n_feat, &n_gau) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    E_INFO("Reading source transition matrices: %s\n", src_tmat_path);
    if (s3tmat_read(src_tmat_path, &src_tmat, &n_tmat_src, &n_state_pm) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    E_INFO("Reading source means: %s\n", src_mean_path);
    if (s3gau_read(src_mean_path, &src_mean, &n_cb_src, &tmp_n_feat, &tmp_n_gau, &veclen) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }
    if (tmp_n_feat != n_feat || tmp_n_gau != n_gau || n_cb_src != n_cb) {
        E_ERROR("Source mean dimensions mismatch\n");
        ret = -1;
        goto cleanup;
    }

    E_INFO("Reading source variances: %s\n", src_var_path);
    if (s3gau_read(src_var_path, &src_var, &n_cb_src, &tmp_n_feat, &tmp_n_gau, &tmp_veclen) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }
    if (tmp_n_feat != n_feat || tmp_n_gau != n_gau || n_cb_src != n_cb) {
        E_ERROR("Source variance dimensions mismatch\n");
        ret = -1;
        goto cleanup;
    }
    for (i = 0; i < n_feat; i++) {
        if (veclen[i] != tmp_veclen[i]) {
            E_ERROR("Variance veclen mismatch\n");
            ret = -1;
            goto cleanup;
        }
    }
    ckd_free(tmp_veclen);
    tmp_veclen = NULL;

    E_INFO("Reading destination model definition: %s\n", dest_mdef_path);
    if (model_def_read(&dest_mdef, dest_mdef_path) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    /* Set up ts2cb mapping for destination */
    if (continuous) {
        E_INFO("Generating continuous ts2cb mapping for destination\n");
        dest_mdef->cb = cont_ts2cb(dest_mdef->n_tied_state);
        n_ts = dest_mdef->n_tied_state;
        n_cb = dest_mdef->n_tied_state;
    } else {
        E_INFO("Generating semi-continuous ts2cb mapping for destination\n");
        dest_mdef->cb = semi_ts2cb(dest_mdef->n_tied_state);
        n_ts = dest_mdef->n_tied_state;
        n_cb = 1;
    }

    E_INFO("Initializing destination model parameters\n");

    n_tmat_dest = dest_mdef->n_tied_tmat;
    init_tmat_dest_list = init_was_added(n_tmat_dest);

    dest_tmat = (float32 ***)ckd_calloc_3d(n_tmat_dest, n_state_pm-1, n_state_pm, sizeof(float32));

    n_mixw_dest = dest_mdef->n_tied_state;
    init_mixw_dest_list = init_was_added(n_mixw_dest);

    dest_mixw = (float32 ***)ckd_calloc_3d(n_mixw_dest, n_feat, n_gau, sizeof(float32));

    /* Calculate number of codebooks for destination */
    for (i = 0, n_cb_dest = 0; i < n_mixw_dest; i++) {
        if (dest_mdef->cb[i] != -1 && (uint32)dest_mdef->cb[i] > n_cb_dest) {
            n_cb_dest = dest_mdef->cb[i];
        }
    }
    ++n_cb_dest;

    init_cb_dest_list = init_was_added(n_cb_dest);

    dest_mean = gauden_alloc_param(n_cb_dest, n_feat, n_gau, veclen);
    dest_var = gauden_alloc_param(n_cb_dest, n_feat, n_gau, veclen);

    E_INFO("Mapping %u source models to %u destination models (%u tied states)\n",
           src_mdef->n_defn, dest_mdef->n_defn, n_mixw_dest);

    /* Map parameters from source to destination */
    for (dest_m = 0; dest_m < dest_mdef->n_defn; dest_m++) {
        dest_m_name = acmod_set_id2name(dest_mdef->acmod_set, dest_m);
        src_m = acmod_set_name2id(src_mdef->acmod_set, dest_m_name);

        if (src_m == NO_ACMOD) {
            /* No exact match - try base phone */
            dest_m_base = acmod_set_base_phone(dest_mdef->acmod_set, dest_m);
            dest_m_base_name = acmod_set_id2name(dest_mdef->acmod_set, dest_m_base);
            src_m_base = acmod_set_name2id(src_mdef->acmod_set, dest_m_base_name);

            if (src_m_base == NO_ACMOD) {
                /* No match at all - use uniform */
                init_uniform_mixw(dest_mixw, &dest_mdef->defn[dest_m], n_feat, n_gau);
            } else {
                /* Use base phone */
                init_model_params(dest_mixw, dest_mean, dest_var, dest_tmat,
                                  &dest_mdef->defn[dest_m], dest_mdef->cb, dest_mdef->acmod_set,
                                  src_mixw, src_mean, src_var, src_tmat,
                                  &src_mdef->defn[src_m_base], src_mdef->cb, src_mdef->acmod_set,
                                  n_feat, n_gau, n_state_pm, veclen);
            }
        } else {
            /* Exact match found */
            init_model_params(dest_mixw, dest_mean, dest_var, dest_tmat,
                              &dest_mdef->defn[dest_m], dest_mdef->cb, dest_mdef->acmod_set,
                              src_mixw, src_mean, src_var, src_tmat,
                              &src_mdef->defn[src_m], src_mdef->cb, src_mdef->acmod_set,
                              n_feat, n_gau, n_state_pm, veclen);
        }
    }

    /* Refuse to write a model with uninitialized states or codebooks. */
    for (m = 0; m < n_mixw_dest; m++) {
        if (init_mixw_dest_list[m] == NULL) {
            E_ERROR("Destination state %u has not been initialized\n", m);
            ret = -1;
        }
    }
    for (m = 0; m < n_cb_dest; m++) {
        if (init_cb_dest_list[m] == NULL) {
            E_ERROR("Destination codebook %u has not been initialized\n", m);
            ret = -1;
        }
    }
    if (ret != 0)
        goto cleanup;

    /* Initialize uninitialized transition matrices from source */
    if (src_tmat && n_tmat_src > 0) {
        uint32 tmat_m, tmat_i, tmat_j;
        for (tmat_m = 0; tmat_m < n_tmat_dest; tmat_m++) {
            if (init_tmat_dest_list[tmat_m] == NULL) {
                for (tmat_i = 0; tmat_i < n_state_pm-1; tmat_i++) {
                    for (tmat_j = 0; tmat_j < n_state_pm; tmat_j++) {
                        dest_tmat[tmat_m][tmat_i][tmat_j] = src_tmat[0][tmat_i][tmat_j];
                    }
                }
            }
        }
    }

    /* Write destination model files */
    E_INFO("Writing destination transition matrices: %s\n", dest_tmat_path);
    if (s3tmat_write(dest_tmat_path, dest_tmat, n_tmat_dest, n_state_pm) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    E_INFO("Writing destination mixture weights: %s\n", dest_mixw_path);
    if (s3mixw_write(dest_mixw_path, dest_mixw, n_mixw_dest, n_feat, n_gau) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    E_INFO("Writing destination means: %s\n", dest_mean_path);
    if (s3gau_write(dest_mean_path, (const vector_t ***)dest_mean, n_cb_dest, n_feat, n_gau, veclen) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    E_INFO("Writing destination variances: %s\n", dest_var_path);
    if (s3gau_write(dest_var_path, (const vector_t ***)dest_var, n_cb_dest, n_feat, n_gau, veclen) != S3_SUCCESS) {
        ret = -1;
        goto cleanup;
    }

    E_INFO("Initialized CD model: %u tied states, %u codebooks\n", n_mixw_dest, n_cb_dest);

cleanup:
    if (veclen) ckd_free(veclen);
    if (tmp_veclen) ckd_free(tmp_veclen);
    if (src_mean) gauden_free_param(src_mean);
    if (src_var) gauden_free_param(src_var);
    if (dest_mean) gauden_free_param(dest_mean);
    if (dest_var) gauden_free_param(dest_var);
    if (src_mdef) model_def_free(src_mdef);
    if (dest_mdef) model_def_free(dest_mdef);
    if (src_mixw) ckd_free_3d(src_mixw);
    if (dest_mixw) ckd_free_3d(dest_mixw);
    if (src_tmat) ckd_free_3d(src_tmat);
    if (dest_tmat) ckd_free_3d(dest_tmat);

    /* Note: init_*_dest_list arrays are leaked here. For a proper implementation,
     * we'd need to track and free them. */

    return ret;
}
